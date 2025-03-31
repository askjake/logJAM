#!/usr/bin/env python3
import os
import sys
import io
import json
import argparse
import datetime
import numpy as np
import psycopg2
from psycopg2.extras import DictCursor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------------
# 1) Model Definition
# -------------------------------------------------------------------------
class LogAutoencoder(nn.Module):
    def __init__(self, input_size, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_size)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# -------------------------------------------------------------------------
# 2) Create / Load Model from DB
# -------------------------------------------------------------------------
def ensure_enrichdb_table(pg_conn, enrichdb_table="enrichdb"):
    """
    Creates (if needed) a table to store multiple autoencoder models, keyed by (input_dim, latent_dim).
    Also ensures columns exist and there's a unique index on (input_dim, latent_dim).
    """
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS "{enrichdb_table}" (
      input_dim INT NOT NULL,
      latent_dim INT NOT NULL,
      model_data BYTEA NOT NULL,
      last_updated TIMESTAMP NOT NULL
    );
    """
    with pg_conn.cursor() as cur:
        cur.execute(create_sql)

    required_columns = {
        "input_dim": "INT NOT NULL",
        "latent_dim": "INT NOT NULL",
        "model_data": "BYTEA NOT NULL",
        "last_updated": "TIMESTAMP NOT NULL"
    }
    with pg_conn.cursor() as cur:
        for col_name, col_def in required_columns.items():
            alter_sql = f"""
            ALTER TABLE "{enrichdb_table}"
            ADD COLUMN IF NOT EXISTS {col_name} {col_def};
            """
            cur.execute(alter_sql)

    # Unique index for ON CONFLICT
    with pg_conn.cursor() as cur:
        idx_name = f"{enrichdb_table}_uniq_idx"
        create_idx_sql = f"""
        CREATE UNIQUE INDEX IF NOT EXISTS {idx_name}
        ON "{enrichdb_table}" (input_dim, latent_dim);
        """
        cur.execute(create_idx_sql)

    pg_conn.commit()

def load_model_from_db(pg_conn, input_dim, latent_dim, device="cpu", enrichdb_table="enrichdb"):
    select_sql = f"""
    SELECT model_data
    FROM "{enrichdb_table}"
    WHERE input_dim = %s
      AND latent_dim = %s
    LIMIT 1
    """
    with pg_conn.cursor() as cur:
        cur.execute(select_sql, (input_dim, latent_dim))
        row = cur.fetchone()
        if not row:
            return None

        model_bytes = row[0]
        buffer = io.BytesIO(model_bytes)
        model = LogAutoencoder(input_dim, latent_dim=latent_dim)
        try:
            checkpoint = torch.load(buffer, map_location=device)
            model.load_state_dict(checkpoint["state_dict"])
            return model
        except Exception as e:
            print(f"[Autoencoder] Error loading model from DB: {e}")
            return None

def store_model_in_db(pg_conn, model, input_dim, latent_dim, enrichdb_table="enrichdb"):
    checkpoint = {"state_dict": model.state_dict()}
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    model_bytes = buffer.getvalue()

    upsert_sql = f"""
    INSERT INTO "{enrichdb_table}" (input_dim, latent_dim, model_data, last_updated)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (input_dim, latent_dim)
    DO UPDATE SET model_data = EXCLUDED.model_data,
                  last_updated = EXCLUDED.last_updated;
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    with pg_conn.cursor() as cur:
        cur.execute(upsert_sql, (input_dim, latent_dim, psycopg2.Binary(model_bytes), now))
    pg_conn.commit()

# -------------------------------------------------------------------------
# 3) Training
# -------------------------------------------------------------------------
def train_autoencoder(model, train_data, epochs=50, batch_size=32, learning_rate=1e-3, table_name=None):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(dataset)
        if table_name:
            print(f"\r[Autoencoder][{table_name}] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}", end=' ')
        else:
            print(f"\r[Autoencoder] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}", end=' ')
    print()
    return model

# -------------------------------------------------------------------------
# 4) Detect Anomalies
# -------------------------------------------------------------------------
def detect_anomalies(model, data, threshold=None):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(data, dtype=torch.float32)
        reconstructed = model(x)
        mse = ((x - reconstructed) ** 2).mean(dim=1).numpy()

    if threshold is None:
        threshold = np.mean(mse) + 2 * np.std(mse)
    anomalies = mse > threshold
    return anomalies, mse, threshold

# -------------------------------------------------------------------------
# 5) Plot MSE Distribution
# -------------------------------------------------------------------------
def plot_mse_distribution(mse_scores, threshold, table_name, save_path="mse_dist.png"):
    plt.figure(figsize=(8,6))
    plt.hist(mse_scores, bins=50, alpha=0.7, label="Reconstruction MSE")
    plt.axvline(x=threshold, color='red', linestyle='--', label=f"Threshold={threshold:.4f}")
    plt.title(f"MSE Distribution - {table_name}")
    plt.xlabel("MSE")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[Autoencoder] MSE distribution plot saved to {save_path}")

# -------------------------------------------------------------------------
# 6) Data from DB (Keep IDs + unscaled timestamps)
# -------------------------------------------------------------------------
def fetch_table_data_autoencoder(pg_conn, table_name, start_time=None, end_time=None):
    """
    Returns (features, row_ids, unscaled_ts).
      - features: scaled data_len
      - row_ids: primary keys for reliable row updates
      - unscaled_ts: optional, if we want to debug or revert to time-based updates
    """
    where_clauses = ["timestamp IS NOT NULL"]
    params = {}
    if start_time:
        where_clauses.append("timestamp >= %(start_time)s")
        params["start_time"] = start_time
    if end_time:
        where_clauses.append("timestamp <= %(end_time)s")
        params["end_time"] = end_time

    where_sql = " AND ".join(where_clauses)

    sql_query = f"""
    SELECT id,
           EXTRACT(EPOCH FROM timestamp) AS ts_epoch,
           COALESCE(LENGTH(data), 0) AS data_len
    FROM "{table_name}"
    WHERE {where_sql}
    ORDER BY timestamp ASC
    """
    with pg_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(sql_query, params)
        rows = cur.fetchall()
    if not rows:
        return np.array([]), [], []

    row_ids = []
    unscaled_ts = []
    data_lens = []
    for r in rows:
        row_ids.append(r["id"])
        unscaled_ts.append(float(r["ts_epoch"] or 0.0))
        data_lens.append(float(r["data_len"] or 0.0))

    # scale data_len only
    data_lens = np.array(data_lens, dtype=np.float32).reshape(-1,1)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_lens)

    return scaled_data, row_ids, unscaled_ts

# -------------------------------------------------------------------------
# 7) Update Table for Anomalies (Use id-based matching!)
# -------------------------------------------------------------------------
def ensure_anomaly_columns(pg_conn, table_name):
    alter_sqls = [
        f'ALTER TABLE "{table_name}" ADD COLUMN IF NOT EXISTS autoenc_score DOUBLE PRECISION',
        f'ALTER TABLE "{table_name}" ADD COLUMN IF NOT EXISTS autoenc_threshold DOUBLE PRECISION',
        f'ALTER TABLE "{table_name}" ADD COLUMN IF NOT EXISTS autoenc_is_anomaly BOOLEAN'
    ]
    with pg_conn.cursor() as cur:
        for sql_cmd in alter_sqls:
            cur.execute(sql_cmd)
    pg_conn.commit()

def store_anomalies_in_table(pg_conn, table_name, row_ids, anomalies, mse_scores, threshold):
    """
    Updates each row using the row's primary key (id). This ensures no float/timestamp mismatch.
    """
    update_sql = f"""
    UPDATE "{table_name}"
    SET autoenc_score = %s,
        autoenc_threshold = %s,
        autoenc_is_anomaly = %s
    WHERE id = %s
    """

    with pg_conn.cursor() as cur:
        for i, rid in enumerate(row_ids):
            is_anom = anomalies[i]
            score_val = mse_scores[i]
            cur.execute(
                update_sql,
                (
                    float(score_val),
                    float(threshold),
                    bool(is_anom),
                    rid
                )
            )
    pg_conn.commit()

# -------------------------------------------------------------------------
# 8) Single Table Analysis
# -------------------------------------------------------------------------
def run_autoencoder_analysis(pg_conn,
                             table_name,
                             enrichdb_table="enrichdb",
                             latent_dim=None,
                             epochs=10,
                             batch_size=32,
                             start_time=None,
                             end_time=None,
                             plot=False):

    ensure_enrichdb_table(pg_conn, enrichdb_table)

    features, row_ids, unscaled_ts = fetch_table_data_autoencoder(pg_conn, table_name, start_time, end_time)
    if features.size == 0:
        print(f"[Autoencoder][{table_name}] No data found. Skipping.")
        return

    input_size = features.shape[1]
    if latent_dim is None:
        latent_dim = min(16, max(2, input_size // 2))

    model = load_model_from_db(pg_conn, input_size, latent_dim, enrichdb_table=enrichdb_table)
    if not model:
        print(f"[Autoencoder][{table_name}] Creating new model (dim={input_size}, latent_dim={latent_dim}).")
        model = LogAutoencoder(input_size, latent_dim=latent_dim)

    model = train_autoencoder(model, features, epochs=epochs, batch_size=batch_size, table_name=table_name)

    anomalies, mse_scores, threshold = detect_anomalies(model, features)
    print(f"[Autoencoder][{table_name}] => threshold={threshold:.4f}, #Anomalies={np.sum(anomalies)}")

    ensure_anomaly_columns(pg_conn, table_name)
    store_anomalies_in_table(pg_conn, table_name, row_ids, anomalies, mse_scores, threshold)

    if plot:
        out_plot = f"mse_dist_{table_name}.png"
        plot_mse_distribution(mse_scores, threshold, table_name, out_plot)

    store_model_in_db(pg_conn, model, input_size, latent_dim, enrichdb_table=enrichdb_table)

# -------------------------------------------------------------------------
# 9) All tables
# -------------------------------------------------------------------------
def get_all_user_tables(pg_conn):
    query = """
    SELECT tablename
    FROM pg_catalog.pg_tables
    WHERE schemaname = 'public'
      AND tablename ~ '^R[0-9]{10}$'
    """
    with pg_conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
    return [r[0] for r in rows]

# -------------------------------------------------------------------------
# 10) Main CLI
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Autoencoder-based Anomaly Detection (DB-stored models).")
    parser.add_argument("--table_name", type=str, help="Analyze only this table. Otherwise, all user tables.")
    parser.add_argument("--enrichdb_table", type=str, default=None,
                        help="Name of DB table that stores the autoencoder models.")
    parser.add_argument("--latent_dim", type=int, default=2, help="Latent dimension for the autoencoder.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--start_time", type=str, default=None, help="Optional start time (ISO format).")
    parser.add_argument("--end_time", type=str, default=None, help="Optional end time.")
    parser.add_argument("--plot", action="store_true", help="If set, generate an MSE distribution plot for each table.")

    args = parser.parse_args()

    if args.enrichdb_table is None:
        args.enrichdb_table = f"encoded_master{args.latent_dim}"

    credentials_file = "credentials.txt"
    if not os.path.exists(credentials_file):
        print(f"❌ credentials.txt not found: {credentials_file}")
        sys.exit(1)

    with open(credentials_file, "r") as f:
        creds = json.load(f)

    db_host = creds.get("DB_HOST")
    db_name = creds.get("DB_NAME")
    db_user = creds.get("DB_USER")
    db_pass = creds.get("DB_PASSWORD")

    try:
        pg_conn = psycopg2.connect(host=db_host, database=db_name, user=db_user, password=db_pass)
    except Exception as e:
        print(f"❌ Error connecting to PostgreSQL: {e}")
        sys.exit(1)

    if args.table_name:
        run_autoencoder_analysis(
            pg_conn,
            table_name=args.table_name,
            enrichdb_table=args.enrichdb_table,
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            start_time=args.start_time,
            end_time=args.end_time,
            plot=args.plot
        )
    else:
        tables = get_all_user_tables(pg_conn)
        for t in tables:
            run_autoencoder_analysis(
                pg_conn,
                table_name=t,
                enrichdb_table=args.enrichdb_table,
                latent_dim=args.latent_dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                start_time=args.start_time,
                end_time=args.end_time,
                plot=args.plot
            )

    pg_conn.close()

if __name__ == "__main__":
    main()
