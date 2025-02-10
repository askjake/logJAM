# analysis/anomaly_detection/lstm_anomaly.py
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import psycopg2
from psycopg2.extras import DictCursor
from psycopg2 import sql


class LogLSTMAnomalyDetector(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LogLSTMAnomalyDetector, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use last time step
        return out

class LogSequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Input: all but last step, Target: last step
        return torch.tensor(seq[:-1], dtype=torch.float32), torch.tensor(seq[-1], dtype=torch.float32)

def train_lstm_anomaly(sequences, input_size, hidden_size=32, num_layers=1, epochs=50, batch_size=32, learning_rate=1e-3):
    dataset = LogSequenceDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LogLSTMAnomalyDetector(input_size, hidden_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, target in dataloader:
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        avg_loss = total_loss / len(dataset)
        #print(f"[LSTM] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    return model

def detect_lstm_anomalies(model, sequences, threshold=None):
    """
    Returns:
      anomalies: a boolean array, True where anomaly is detected
      losses: an array of MSE for each sequence
      threshold: the numeric threshold used
    """
    model.eval()
    losses = []
    with torch.no_grad():
        for seq in sequences:
            seq_tensor = torch.tensor(seq, dtype=torch.float32)
            input_seq = seq_tensor[:-1].unsqueeze(0)  # shape (1, seq_length-1, input_size)
            target = seq_tensor[-1].unsqueeze(0)
            output = model(input_seq)
            loss = nn.functional.mse_loss(output, target, reduction='mean').item()
            losses.append(loss)

    losses = np.array(losses)
    if threshold is None:
        threshold = np.mean(losses) + 2*np.std(losses)
    anomalies = losses > threshold
    return anomalies, losses, threshold

def fetch_sequence_data_lstm(pg_conn, table_name, seq_len=10):
    """
    Example: fetch logs from table_name and produce sequences.
    We'll build a naive 'timestamp + data_len' vector for each row,
    then chunk them into overlapping sequences of length seq_len.

    This logic can be customized to include more features:
      - E.g., categories, stbh, or other numeric fields.
    """
    sql_query = f"""
    SELECT EXTRACT(EPOCH FROM timestamp) AS ts_epoch,
           COALESCE(LENGTH(data), 0) AS data_len
    FROM "{table_name}"
    WHERE timestamp IS NOT NULL
    ORDER BY timestamp ASC
    """
    with pg_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(sql_query)
        rows = cur.fetchall()

    if not rows:
        print(f"[LSTM] No data found in {table_name}.")
        return []

    vectors = []
    for row in rows:
        ts_epoch = float(row["ts_epoch"]) if row["ts_epoch"] else 0.0
        data_len = float(row["data_len"]) if row["data_len"] else 0.0
        vectors.append([ts_epoch, data_len])

    # Build sequences of length seq_len
    sequences = []
    for i in range(len(vectors) - seq_len + 1):
        seq = vectors[i : i+seq_len]
        sequences.append(seq)

    return sequences

def run_lstm_analysis(pg_conn, table_name, seq_len=10, hidden_size=16, epochs=10):
    sequences = fetch_sequence_data_lstm(pg_conn, table_name, seq_len=seq_len)
    if len(sequences) < 2:
        print(f"[LSTM] Skipping analysis: Not enough sequences in {table_name}.")
        return

    input_size = len(sequences[0][0])  # number of features (2 in this example)
    print(f"\n[LSTM] Training on table '{table_name}', #sequences={len(sequences)}, input_size={input_size}")

    model = train_lstm_anomaly(sequences, input_size=input_size, hidden_size=hidden_size, epochs=epochs)
    anomalies, losses, threshold = detect_lstm_anomalies(model, sequences)
    total_anomalies = np.sum(anomalies)
    print(f"[LSTM] Completed analysis on '{table_name}'. Threshold={threshold:.4f}, #Anomalies={total_anomalies}.")

    if total_anomalies > 0:
        print(f"[LSTM] Detailed anomaly info for '{table_name}':")
        # Explanation for your boss/engineer:
        print("   These anomalies indicate sequences whose final step didn't match the model's expectation.")
        print("   Large MSE = the last data point is out of normal range. This can imply unusual time gaps,")
        print("   increased 'data' length, or something else the LSTM didn't learn as 'typical' usage.\n")

        for idx, is_anomaly in enumerate(anomalies):
            if is_anomaly:
                seq = sequences[idx]
                mse_val = losses[idx]
                # Show the last record's info (timestamp, data_len)
                # Because the final step is what the model tried to predict
                last_entry = seq[-1]  # [ts_epoch, data_len]
                timestamp_str = f"{last_entry[0]:.0f} (epoch)"
                data_len_str = f"{last_entry[1]:.0f} (len of 'data')"
                print(f"   Sequence idx {idx} => MSE={mse_val:.4f}   TS={timestamp_str}   data_len={data_len_str}")
                print("   (Engineer can cross-reference this approximate time to logs for deeper analysis.)")
    else:
        print(f"[LSTM] No anomalies detected in '{table_name}'. This suggests log usage is consistent with training data.")

    # Return them if you want to store or query further
    return anomalies, losses, threshold

def get_all_user_tables(pg_conn):
    """
    Returns a list of user-defined table names (i.e. not system tables)
    from the 'public' schema (adjust if needed for other schemas).
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
    tables = [r[0] for r in rows]
    return tables

def main():
    parser = argparse.ArgumentParser(description="LSTM Anomaly Detection on ALL tables (with Explanations)")
    parser.add_argument("--seq_len", type=int, default=10, help="Sequence length for building sequences")
    parser.add_argument("--hidden_size", type=int, default=16, help="Hidden size of the LSTM layer")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    args = parser.parse_args()

    # Determine the base directory that contains credentials.txt
    three_levels_up = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
    credentials_file = os.path.join(three_levels_up, "credentials.txt")

    if not os.path.exists(credentials_file):
        print(f"❌ credentials.txt not found at: {credentials_file}")
        return

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

    all_tables = get_all_user_tables(pg_conn)
    print(f"[LSTM] Found {len(all_tables)} user-defined tables in database '{db_name}'.")

    for table_name in all_tables:
        try:
            run_lstm_analysis(pg_conn, table_name, seq_len=args.seq_len, hidden_size=args.hidden_size, epochs=args.epochs)
        except Exception as e:
            print(f"❌ Error during LSTM analysis for table '{table_name}': {e}")

    pg_conn.close()
    print("[LSTM] Completed analysis on all tables. Connection closed.")

if __name__ == "__main__":
    main()
