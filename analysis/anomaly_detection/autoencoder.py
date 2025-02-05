# analysis/anomaly_detection/autoencoder.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import psycopg2  # For Postgres
from psycopg2.extras import DictCursor

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

def fetch_table_data_autoencoder(pg_conn, table_name):
    """
    Example: fetch logs from table_name and convert to numeric features for autoencoder.
    We'll create a simple 2D feature: 
      1) timestamp as epoch 
      2) length of the 'data' field
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

    # Convert to NumPy
    if not rows:
        print(f"[Autoencoder] No data found in {table_name}.")
        return np.array([])

    features = []
    for row in rows:
        ts_epoch = float(row["ts_epoch"]) if row["ts_epoch"] else 0.0
        data_len = float(row["data_len"]) if row["data_len"] else 0.0
        features.append([ts_epoch, data_len])

    return np.array(features, dtype=np.float32)

def run_autoencoder_analysis(pg_conn, table_name, latent_dim=2, epochs=10, batch_size=32):
    """
    High-level function to fetch data from Postgres, train autoencoder, detect anomalies.
    """
    data = fetch_table_data_autoencoder(pg_conn, table_name)
    if data.size == 0:
        print(f"[Autoencoder] Skipping analysis: No data in {table_name}.")
        return

    input_size = data.shape[1]
    print(f"[Autoencoder] Training on table {table_name}, shape={data.shape}")

    model = train_autoencoder(data, input_size=input_size, latent_dim=latent_dim, epochs=epochs, batch_size=batch_size)
    anomalies, mse_scores, threshold = detect_anomalies(model, data)
    print(f"[Autoencoder] Completed analysis. Threshold={threshold:.4f}, #Anomalies={np.sum(anomalies)}.")

    # Optionally, we could store anomalies in the DB or log them
    # For now, just print a summary
    return anomalies, mse_scores, threshold
