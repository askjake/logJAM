# analysis/anomaly_detection/autoencoder.py

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime

class LogAutoencoder(nn.Module):
    def __init__(self, input_size, latent_dim=16):
        super(LogAutoencoder, self).__init__()
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

def train_autoencoder(train_data, input_size, latent_dim=16, epochs=50, batch_size=32, learning_rate=1e-3):
    """Trains the autoencoder on the provided train_data (NumPy array)."""
    model = LogAutoencoder(input_size, latent_dim)
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
        print(f"[Autoencoder] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    return model

def detect_anomalies(model, data, threshold=None):
    """
    Uses the trained model to detect anomalies in the data.
    :param model: Trained LogAutoencoder
    :param data: NumPy array of shape (num_samples, input_size)
    :param threshold: If None, set to mean + 2*std of reconstruction errors.
    :return: (anomalies, mse_scores, threshold)
    """
    model.eval()
    with torch.no_grad():
        x = torch.tensor(data, dtype=torch.float32)
        reconstructed = model(x)
        mse = ((x - reconstructed)**2).mean(dim=1).numpy()

    if threshold is None:
        threshold = np.mean(mse) + 2*np.std(mse)

    anomalies = mse > threshold
    return anomalies, mse, threshold

def fetch_table_data_autoencoder(pg_conn, table_name, start_time=None, end_time=None):
    """
    Fetch logs from table_name and convert to numeric features for autoencoder:
      1) timestamp as epoch
      2) length of the 'data' field

    If start_time/end_time are provided (as ISO strings), filter logs to that range.
    This helps match the timeframe of a user's complaint.
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

    sql = f"""
    SELECT EXTRACT(EPOCH FROM timestamp) AS ts_epoch,
           COALESCE(LENGTH(data), 0) AS data_len
    FROM "{table_name}"
    WHERE {where_sql}
    ORDER BY timestamp ASC
    """

    with pg_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    if not rows:
        print(f"[Autoencoder] No data found in {table_name} (with filters).")
        return np.array([])

    features = []
    for row in rows:
        ts_epoch = float(row["ts_epoch"]) if row["ts_epoch"] else 0.0
        data_len = float(row["data_len"]) if row["data_len"] else 0.0
        features.append([ts_epoch, data_len])

    return np.array(features, dtype=np.float32)

def run_autoencoder_analysis(pg_conn, table_name, latent_dim=2, epochs=10, batch_size=32,
                             start_time=None, end_time=None):
    """
    High-level function to fetch data from Postgres, train autoencoder, detect anomalies.
    """
    data = fetch_table_data_autoencoder(pg_conn, table_name, start_time, end_time)
    if data.size == 0:
        print(f"[Autoencoder] Skipping: No data in {table_name} (with filters).")
        return

    input_size = data.shape[1]
    print(f"[Autoencoder] Training on table {table_name}, shape={data.shape}")

    model = train_autoencoder(data, input_size=input_size, latent_dim=latent_dim,
                              epochs=epochs, batch_size=batch_size)
    anomalies, mse_scores, threshold = detect_anomalies(model, data)
    print(f"[Autoencoder] Completed analysis. Threshold={threshold:.4f}, #Anomalies={np.sum(anomalies)}.")
    # Optionally store anomalies back in DB or log them
    return anomalies, mse_scores, threshold

# ---------- ADDED MAIN FOR STANDALONE USAGE -------------
def get_all_user_tables(pg_conn):
    """
    Returns a list of user-defined table names in the 'public' schema,
    ignoring system tables. Adjust if you store logs in a different schema.
    """
    query = """
    SELECT tablename
    FROM pg_catalog.pg_tables
    WHERE schemaname = 'public'
      AND tablename NOT LIKE 'pg_%'
      AND tablename NOT LIKE 'sql_%';
    """
    with pg_conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
    return [r[0] for r in rows]

def main():
    parser = argparse.ArgumentParser(description="Autoencoder-based Anomaly Detection")
    parser.add_argument("--table_name", type=str,
                        help="If provided, analyze only this table. Otherwise, analyze all user tables.")
    parser.add_argument("--latent_dim", type=int, default=2,
                        help="Latent dimension for the autoencoder.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size.")
    parser.add_argument("--start_time", type=str, default=None,
                        help="Optional start time (ISO format) to filter logs (e.g. '2023-08-01T00:00:00').")
    parser.add_argument("--end_time", type=str, default=None,
                        help="Optional end time (ISO format) to filter logs.")
    args = parser.parse_args()

    # Find the project base dir (two or three levels up from this file)
    base_dir = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
    credentials_file = os.path.join(base_dir, "credentials.txt")

    if not os.path.exists(credentials_file):
        print(f"❌ credentials.txt not found at: {credentials_file}")
        return

    # Load credentials
    try:
        with open(credentials_file, "r") as f:
            creds = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error decoding JSON in {credentials_file}: {e}")
        return

    db_host = creds.get("DB_HOST")
    db_name = creds.get("DB_NAME")
    db_user = creds.get("DB_USER")
    db_pass = creds.get("DB_PASSWORD")

    # Connect to PostgreSQL
    try:
        pg_conn = psycopg2.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_pass
        )
    except Exception as e:
        print(f"❌ Error connecting to PostgreSQL at {db_host}: {e}")
        return

    if args.table_name:
        # Analyze only the specified table
        print(f"[Autoencoder] Analyzing table '{args.table_name}' only.")
        run_autoencoder_analysis(
            pg_conn,
            table_name=args.table_name,
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            start_time=args.start_time,
            end_time=args.end_time
        )
    else:
        # Analyze ALL user tables
        all_tables = get_all_user_tables(pg_conn)
        print(f"[Autoencoder] Found {len(all_tables)} user-defined tables in '{db_name}'.")
        for t in all_tables:
            print(f"\n[Autoencoder] Analyzing table '{t}' ...")
            run_autoencoder_analysis(
                pg_conn,
                table_name=t,
                latent_dim=args.latent_dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                start_time=args.start_time,
                end_time=args.end_time
            )

    pg_conn.close()
    print("[Autoencoder] Completed analysis. Connection closed.")

if __name__ == "__main__":
    main()
