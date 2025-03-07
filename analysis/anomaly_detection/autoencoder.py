"""
autoencoder.py (Refactored)

1) Load or create a model, verifying dimension from the last checkpoint
2) Train (or continue training)
3) Detect anomalies
4) Update the *same* table with columns:
     autoenc_score FLOAT,
     autoenc_threshold FLOAT,
     autoenc_is_anomaly BOOLEAN
5) Optionally plot MSE distribution vs. threshold
6) If dimension mismatch occurs, build a fresh model
"""

import os
import json
import argparse
import numpy as np
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import psycopg2
from psycopg2.extras import DictCursor
import matplotlib.pyplot as plt  # For plotting distribution

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
# 2) Training
# -------------------------------------------------------------------------
def train_autoencoder(model, train_data, epochs=50, batch_size=32, learning_rate=1e-3):
    """Continue training model on train_data (NumPy array)."""
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
        print(f"\r[Autoencoder] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}", end='')
    print()  # newline after final epoch
    return model

# -------------------------------------------------------------------------
# 3) Detect Anomalies
# -------------------------------------------------------------------------
def detect_anomalies(model, data, threshold=None):
    """Return (anomalies_bool, mse_scores, threshold) for each row."""
    model.eval()
    with torch.no_grad():
        x = torch.tensor(data, dtype=torch.float32)
        reconstructed = model(x)
        mse = ((x - reconstructed) ** 2).mean(dim=1).numpy()

    if threshold is None:
        threshold = np.mean(mse) + 2*np.std(mse)

    anomalies = mse > threshold
    return anomalies, mse, threshold

# -------------------------------------------------------------------------
# 4) Plot MSE Distribution
# -------------------------------------------------------------------------
def plot_mse_distribution(mse_scores, threshold, table_name, save_path="mse_dist.png"):
    """Plot histogram of MSE and show threshold line."""
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
# 5) Data from DB
# -------------------------------------------------------------------------
def fetch_table_data_autoencoder(pg_conn, table_name, start_time=None, end_time=None):
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
    SELECT EXTRACT(EPOCH FROM timestamp) AS ts_epoch,
           COALESCE(LENGTH(data), 0) AS data_len
    FROM "{table_name}"
    WHERE {where_sql}
    ORDER BY timestamp ASC
    """
    with pg_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(sql_query, params)
        rows = cur.fetchall()

    if not rows:
        print(f"[Autoencoder] No data found in '{table_name}' with filters.")
        return np.array([])

    vectors = []
    for r in rows:
        ts_epoch = float(r["ts_epoch"]) if r["ts_epoch"] else 0.0
        data_len = float(r["data_len"]) if r["data_len"] else 0.0
        vectors.append([ts_epoch, data_len])
    return np.array(vectors, dtype=np.float32)

# -------------------------------------------------------------------------
# 6) Update Table for Anomalies
# -------------------------------------------------------------------------
def ensure_anomaly_columns(pg_conn, table_name):
    """
    Add columns to store autoencoder anomaly context:
      - autoenc_score FLOAT
      - autoenc_threshold FLOAT
      - autoenc_is_anomaly BOOLEAN
    """
    alter_sqls = [
        f'ALTER TABLE "{table_name}" ADD COLUMN IF NOT EXISTS autoenc_score DOUBLE PRECISION',
        f'ALTER TABLE "{table_name}" ADD COLUMN IF NOT EXISTS autoenc_threshold DOUBLE PRECISION',
        f'ALTER TABLE "{table_name}" ADD COLUMN IF NOT EXISTS autoenc_is_anomaly BOOLEAN'
    ]
    with pg_conn.cursor() as cur:
        for sql_cmd in alter_sqls:
            cur.execute(sql_cmd)
    pg_conn.commit()

def store_anomalies_in_table(pg_conn, table_name, data, anomalies, mse_scores, threshold):
    """
    For each row in `data` => [ts_epoch, data_len],
    update row in <table_name> where EXTRACT(EPOCH FROM timestamp) ~ ts_epoch,
    setting autoenc_score, autoenc_threshold, autoenc_is_anomaly.
    """
    update_sql = f"""
    UPDATE "{table_name}"
    SET autoenc_score = %s,
        autoenc_threshold = %s,
        autoenc_is_anomaly = %s
    WHERE
       ABS(EXTRACT(EPOCH FROM timestamp) - %s) < 0.001
    """

    with pg_conn.cursor() as cur:
        for i, is_anom in enumerate(anomalies):
            ts_epoch = data[i][0]           # from e.g. [ts_epoch, data_len]
            score_val = mse_scores[i]
            # Convert everything to pure Python
            cur.execute(
                update_sql,
                (
                    float(score_val),        # python float
                    float(threshold),        # python float
                    bool(is_anom),           # python bool
                    float(ts_epoch),         # python float
                )
            )
    pg_conn.commit()


# -------------------------------------------------------------------------
# 7) Single Table Analysis
# -------------------------------------------------------------------------
def run_autoencoder_analysis(pg_conn, table_name,
                             model_path=None,
                             latent_dim=2,
                             epochs=10,
                             batch_size=32,
                             start_time=None,
                             end_time=None,
                             plot=False):
    """
    1) Load existing autoencoder .pt (check dimension) if model_path
    2) Fetch data
    3) Possibly train or continue training
    4) Detect anomalies, store them in the same table (with additional columns)
    5) Save model
    6) Optionally plot MSE distribution
    """
    data = fetch_table_data_autoencoder(pg_conn, table_name, start_time, end_time)
    if data.size == 0:
        print(f"[Autoencoder] Skipping {table_name}, no data.")
        return

    input_size = data.shape[1]
    print(f"[Autoencoder] Table '{table_name}' => shape={data.shape}")

    # create or load model
    model = LogAutoencoder(input_size, latent_dim=latent_dim)
    loaded_model_shape = None

    if model_path and os.path.exists(model_path):
        print(f"[Autoencoder] Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        # Optionally store "input_size" inside checkpoint to avoid mismatch
        if "model_input_dim" in checkpoint:
            loaded_model_shape = checkpoint["model_input_dim"]
            if loaded_model_shape != input_size:
                print(f"[Autoencoder] ❌ Dimension mismatch: loaded {loaded_model_shape}, current {input_size}")
                print("[Autoencoder] Starting a fresh model instead.")
            else:
                # load state
                model.load_state_dict(checkpoint["state_dict"])
                print("[Autoencoder] Successfully loaded existing model.")
        else:
            # if no dimension metadata, let's try a raw load
            try:
                model.load_state_dict(checkpoint)
                print("[Autoencoder] Successfully loaded model (no dimension check).")
            except RuntimeError as e:
                print(f"[Autoencoder] ❌ Dimension mismatch or load error: {e}\nStarting fresh model.")

    # train
    model = train_autoencoder(model, train_data=data, epochs=epochs, batch_size=batch_size)

    # detect anomalies
    anomalies, mse_scores, threshold = detect_anomalies(model, data)
    print(f"[Autoencoder] => #Anomalies={np.sum(anomalies)}  threshold={threshold:.4f}")

    # add columns in the same table
    ensure_anomaly_columns(pg_conn, table_name)
    store_anomalies_in_table(pg_conn, table_name, data, anomalies, mse_scores, threshold)

    # Optionally plot MSE distribution
    if plot:
        out_plot = f"mse_dist_{table_name}.png"
        plot_mse_distribution(mse_scores, threshold, table_name, out_plot)

    # Save model
    if model_path:
        print(f"[Autoencoder] Saving model to {model_path}")
        # store dimension inside the checkpoint
        checkpoint = {
            "model_input_dim": input_size,
            "state_dict": model.state_dict()
        }
        torch.save(checkpoint, model_path)

# -------------------------------------------------------------------------
# 8) All tables
# -------------------------------------------------------------------------
def get_all_user_tables(pg_conn):
    """
    Returns only tables where the name is 'R' followed by exactly 10 digits.
    """
    query = """
    SELECT tablename
    FROM pg_catalog.pg_tables
    WHERE schemaname = 'public'
      AND tablename ~ '^R[0-9]{10}$'  -- Regex to match "R" + 10 digits
    """
    with pg_conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
    return [r[0] for r in rows]


# -------------------------------------------------------------------------
# 9) Main CLI
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Autoencoder-based Anomaly Detection w/ model checkpointing, dimension check, and inline anomalies")
    parser.add_argument("--table_name", type=str,
                        help="Analyze only this table. Otherwise, all user tables.")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to .pt checkpoint to load/save model, e.g. 'autoenc_model_2d.pt'.")
    parser.add_argument("--latent_dim", type=int, default=2,
                        help="Latent dimension for the autoencoder.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--start_time", type=str, default=None,
                        help="Optional start time (ISO format).")
    parser.add_argument("--end_time", type=str, default=None,
                        help="Optional end time.")
    parser.add_argument("--plot", action="store_true",
                        help="If set, generate an MSE distribution plot for each table.")

    args = parser.parse_args()

    credentials_file = "credentials.txt"
    if not os.path.exists(credentials_file):
        print(f"❌ credentials.txt not found: {credentials_file}")
        return

    with open(credentials_file, "r") as f:
        creds = json.load(f)

    db_host = creds.get("DB_HOST")
    db_name = creds.get("DB_NAME")
    db_user = creds.get("DB_USER")
    db_pass = creds.get("DB_PASSWORD")

    import psycopg2
    try:
        pg_conn = psycopg2.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_pass
        )
    except Exception as e:
        print(f"❌ Error connecting to PostgreSQL: {e}")
        return

    if args.table_name:
        run_autoencoder_analysis(
            pg_conn,
            table_name=args.table_name,
            model_path=args.model_path,
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            start_time=args.start_time,
            end_time=args.end_time,
            plot=args.plot
        )
    else:
        tables = get_all_user_tables(pg_conn)
        print(f"[Autoencoder] Found {len(tables)} user tables.")
        for t in tables:
            print(f"\n[Autoencoder] Analyzing table '{t}' ...")
            run_autoencoder_analysis(
                pg_conn,
                table_name=t,
                model_path=args.model_path,
                latent_dim=args.latent_dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                start_time=args.start_time,
                end_time=args.end_time,
                plot=args.plot
            )

    pg_conn.close()
    print("[Autoencoder] Done. Connection closed.")

if __name__ == "__main__":
    main()
