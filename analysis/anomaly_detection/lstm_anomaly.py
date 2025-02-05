# analysis/anomaly_detection/lstm_anomaly.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import psycopg2
from psycopg2.extras import DictCursor

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
        print(f"[LSTM] Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")

    return model

def detect_lstm_anomalies(model, sequences, threshold=None):
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
    """
    sql = f"""
    SELECT EXTRACT(EPOCH FROM timestamp) AS ts_epoch,
           COALESCE(LENGTH(data), 0) AS data_len
    FROM "{table_name}"
    WHERE timestamp IS NOT NULL
    ORDER BY timestamp ASC
    """
    with pg_conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(sql)
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
    print(f"[LSTM] Training on table {table_name}, #sequences={len(sequences)}, input_size={input_size}")

    model = train_lstm_anomaly(sequences, input_size=input_size, hidden_size=hidden_size, epochs=epochs)
    anomalies, losses, threshold = detect_lstm_anomalies(model, sequences)
    print(f"[LSTM] Completed analysis. Threshold={threshold:.4f}, #Anomalies={np.sum(anomalies)}.")

    return anomalies, losses, threshold
