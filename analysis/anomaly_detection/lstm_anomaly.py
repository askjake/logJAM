#!/usr/bin/env python3
"""
lstm_anomaly.py (Refactored)

1) Build or load LSTM model from checkpoint, verifying input dimension.
2) Train (or continue training).
3) Detect anomalies.
4) Store anomalies in the same DB table columns (lstm_score, lstm_threshold, lstm_is_anomaly).
5) Optionally plot final MSE distribution.
"""

import os
import sys
import json
import argparse
import numpy as np
import datetime
import pytz

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import psycopg2
from psycopg2.extras import DictCursor
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# 1) LSTM Model
# -------------------------------------------------------------------------
class LogLSTMAnomalyDetector(nn.Module):
    def __init__(self, input_size, hidden_size=16, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # last time step
        return out

# -------------------------------------------------------------------------
# 2) Sequence Dataset
# -------------------------------------------------------------------------
class LogSequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # input = seq[:-1], target = seq[-1]
        return (torch.tensor(seq[:-1], dtype=torch.float32),
                torch.tensor(seq[-1],  dtype=torch.float32))

# -------------------------------------------------------------------------
# 3) LSTM Training
# -------------------------------------------------------------------------
def train_lstm_anomaly(model, sequences, epochs=10, batch_size=32, learning_rate=1e-3):
    dataset    = LogSequenceDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, target in dataloader:
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"[LSTM] Epoch {epoch+1}/{epochs}, loss={avg_loss:.4f}")
    return model

# -------------------------------------------------------------------------
# 4) Anomaly Detection
# -------------------------------------------------------------------------
def detect_lstm_anomalies(model, sequences, threshold=None):
    model.eval()
    losses = []
    with torch.no_grad():
        for seq in sequences:
            seq_tensor = torch.tensor(seq, dtype=torch.float32)
            input_seq  = seq_tensor[:-1].unsqueeze(0)
            target     = seq_tensor[-1].unsqueeze(0)
            out        = model(input_seq)
            mse_loss   = nn.functional.mse_loss(out, target, reduction='mean').item()
            losses.append(mse_loss)
    losses = np.array(losses)
    if threshold is None:
        threshold = np.mean(losses) + 2.0 * np.std(losses)
    anomalies = losses > threshold
    return anomalies, losses, threshold

# -------------------------------------------------------------------------
# 5) Build or Load Feature Vectors
# -------------------------------------------------------------------------
def build_feature_vectors(pg_conn, table_name, seq_len=10):
    """
    Build feature vectors using dynamic columns (e.g., ts_epoch, data_len, event_type, etc.)
    from the specified table.
    """
    # Determine which columns exist
    existing_cols = get_table_columns(pg_conn, table_name)

    select_parts = ["EXTRACT(EPOCH FROM timestamp) AS ts_epoch"]
    if 'data' in existing_cols:
        select_parts.append("COALESCE(LENGTH(data), 0) AS data_len")
    else:
        select_parts.append("0 AS data_len")

    if 'event_type' in existing_cols:
        select_parts.append("COALESCE(event_type, 'UNKNOWN') AS event_type")
    else:
        select_parts.append("'UNKNOWN' AS event_type")

    if 'process' in existing_cols:
        select_parts.append("COALESCE(process, 'UNKNOWN') AS process_name")
    else:
        select_parts.append("'UNKNOWN' AS process_name")

    if 'file_line' in existing_cols:
        select_parts.append("COALESCE(file_line, '0') AS file_line")
    else:
        select_parts.append("'0' AS file_line")

    if 'function' in existing_cols:
        select_parts.append("COALESCE(function, '') AS function")
    else:
        select_parts.append("'' AS function")

    select_clause = ", ".join(select_parts)
    sql_query = f"""
    SELECT {select_clause}
    FROM "{table_name}"
    WHERE timestamp IS NOT NULL
    ORDER BY timestamp ASC
    """
    with pg_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(sql_query)
        rows = cur.fetchall()
    if not rows:
        return []

    event_type_dict = {}
    process_list = []

    raw_vectors = []
    for r in rows:
        ts_epoch = float(r["ts_epoch"])
        data_len = float(r.get("data_len", 0.0))

        et = r.get("event_type", "UNKNOWN")
        if et not in event_type_dict:
            event_type_dict[et] = len(event_type_dict)
        event_idx = float(event_type_dict[et])

        proc = r.get("process_name", "UNKNOWN")
        if proc not in process_list:
            process_list.append(proc)
        process_idx = float(process_list.index(proc))

        try:
            file_line_val = float(r.get("file_line", 0.0) or 0.0)
        except:
            file_line_val = 0.0

        function_val = 0.0  # not using function column value in numeric form

        vector = [
            ts_epoch,
            data_len,
            event_idx,
            process_idx,
            file_line_val,
            function_val
        ]
        raw_vectors.append(vector)

    # Build overlapping sequences
    sequences = []
    for i in range(len(raw_vectors) - seq_len + 1):
        seq = raw_vectors[i:i+seq_len]
        sequences.append(seq)
    return sequences

def get_table_columns(pg_conn, table_name):
    query = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = %s
    """
    with pg_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(query, (table_name.lower(),))
        rows = cur.fetchall()
    return set(r["column_name"].lower() for r in rows)

# -------------------------------------------------------------------------
# 6) Update Table Columns for LSTM Anomalies
# -------------------------------------------------------------------------
def ensure_lstm_columns(pg_conn, table_name):
    alter_cmds = [
        f'ALTER TABLE "{table_name}" ADD COLUMN IF NOT EXISTS lstm_score DOUBLE PRECISION',
        f'ALTER TABLE "{table_name}" ADD COLUMN IF NOT EXISTS lstm_threshold DOUBLE PRECISION',
        f'ALTER TABLE "{table_name}" ADD COLUMN IF NOT EXISTS lstm_is_anomaly BOOLEAN'
    ]
    with pg_conn.cursor() as cur:
        for cmd in alter_cmds:
            cur.execute(cmd)
    pg_conn.commit()

def store_lstm_anomalies(pg_conn, table_name, sequences, anomalies, losses, threshold, seq_len=10):
    update_sql = f"""
    UPDATE "{table_name}"
    SET lstm_score = %s,
        lstm_threshold = %s,
        lstm_is_anomaly = %s
    WHERE
        ABS(EXTRACT(EPOCH FROM timestamp) - %s) < 0.001
    """
    try:
        with pg_conn.cursor() as cur:
            for i, is_anom in enumerate(anomalies):
                seq = sequences[i]
                last_row = seq[-1]
                ts_epoch = float(last_row[0])
                score_val = float(losses[i])
                cur.execute(update_sql, (
                    score_val,
                    float(threshold),
                    bool(is_anom),
                    ts_epoch
                ))
        pg_conn.commit()
    except psycopg2.Error as e:
        print(f"[LSTM] ❌ Error updating anomalies in {table_name}: {e}")
        pg_conn.rollback()

# -------------------------------------------------------------------------
# 7) Plot MSE Distribution
# -------------------------------------------------------------------------
def plot_mse_distribution(losses, threshold, table_name, save_path):
    plt.figure(figsize=(8,6))
    plt.hist(losses, bins=50, alpha=0.7, label="LSTM MSE")
    plt.axvline(x=threshold, color='red', linestyle='--', label=f"Threshold={threshold:.4f}")
    plt.title(f"LSTM MSE Distribution - {table_name}")
    plt.xlabel("MSE")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[LSTM] MSE distribution saved to {save_path}")

# -------------------------------------------------------------------------
# 8) LSTM Anomaly Detection
# -------------------------------------------------------------------------
def run_lstm_analysis(pg_conn, table_name, seq_len=10, hidden_size=16,
                      epochs=10, model_path=None, plot=False):
    try:
        sequences = build_feature_vectors(pg_conn, table_name, seq_len=seq_len)
        if len(sequences) < 1:
            print(f"[LSTM] No data for {table_name}, skipping.")
            return

        input_size = len(sequences[0][0])
        model = LogLSTMAnomalyDetector(input_size, hidden_size)

        if model_path and os.path.exists(model_path):
            print(f"[LSTM] Loading model from {model_path} ...")
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint["state_dict"])
            print("[LSTM] Model loaded.")

        model = train_lstm_anomaly(model, sequences, epochs=epochs)
        anomalies, losses, threshold = detect_lstm_anomalies(model, sequences)
        n_anom = np.sum(anomalies)
        print(f"[LSTM] Table '{table_name}': #Anomalies={n_anom}  threshold={threshold:.4f}")

        ensure_lstm_columns(pg_conn, table_name)
        store_lstm_anomalies(pg_conn, table_name, sequences, anomalies, losses, threshold, seq_len=seq_len)

        if plot:
            outpath = f"lstm_mse_{table_name}.png"
            plot_mse_distribution(losses, threshold, table_name, outpath)

        if model_path:
            print(f"[LSTM] Saving model checkpoint => {model_path}")
            checkpoint = {
                "model_input_dim": input_size,
                "state_dict": model.state_dict()
            }
            torch.save(checkpoint, model_path)

    except Exception as e:
        print(f"[LSTM] ❌ Error analyzing {table_name}: {e}")
        pg_conn.rollback()

# -------------------------------------------------------------------------
# 9) MAIN
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


def main():
    parser = argparse.ArgumentParser(description="LSTM-based anomaly detection. Process one table with --table_name, or all tables if not specified.")
    parser.add_argument("--table_name", type=str, help="If set, process only this table.")
    parser.add_argument("--seq_len", type=int, default=10, help="Sequence length for building sequences")
    parser.add_argument("--hidden_size", type=int, default=16, help="Hidden size")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--model_path", type=str, default=None, help="Optional path to load/save LSTM model")
    parser.add_argument("--plot", action="store_true", help="If set, plot MSE distribution")
    args = parser.parse_args()

    cred_file = "credentials.txt"
    if not os.path.exists(cred_file):
        print(f"❌ Missing {cred_file}")
        return

    with open(cred_file) as f:
        creds = json.load(f)

    db_host = creds["DB_HOST"]
    db_name = creds["DB_NAME"]
    db_user = creds["DB_USER"]
    db_pass = creds["DB_PASSWORD"]

    try:
        pg_conn = psycopg2.connect(host=db_host, database=db_name, user=db_user, password=db_pass)
    except Exception as e:
        print(f"❌ DB Connect Error: {e}")
        return

    if args.table_name:
        tables = [args.table_name.strip()]
        print(f"[LSTM] Single table mode: Processing table {tables[0]}")
    else:
        # Process all user-defined tables
        tables = get_all_user_tables(pg_conn)
        print(f"[LSTM] Found {len(tables)} user-defined tables in {db_name}")

    for t in tables:
        print(f"[LSTM] Analyzing table '{t}'")
        try:
            run_lstm_analysis(
                pg_conn, t,
                seq_len=args.seq_len,
                hidden_size=args.hidden_size,
                epochs=args.epochs,
                model_path=args.model_path,
                plot=args.plot
            )
        except Exception as e:
            print(f"[LSTM] Error analyzing {t}: {e}")

    pg_conn.close()
    print("[LSTM] All done.")

if __name__ == "__main__":
    main()
