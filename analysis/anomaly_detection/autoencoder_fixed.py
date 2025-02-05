# analysis/anomaly_detection/autoencoder.py



import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

import numpy as np



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

    """Trains the autoencoder on the provided train_data."""

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

        print(f" Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")  #  Improved logging



    return model





def detect_anomalies(model, data, threshold=None):

    """

    Uses the trained model to detect anomalies in the data.

    :param model: Trained LogAutoencoder

    :param data: NumPy array of shape (num_samples, input_size)

    :param threshold: If not provided, it is set to (mean + 2*std) of reconstruction errors.

    :return: A tuple (anomalies, mse_scores, threshold)

    """

    model.eval()

    with torch.no_grad():

        x = torch.tensor(data, dtype=torch.float32)

        reconstructed = model(x)

        # Calculate mean squared error per sample

        mse = ((x - reconstructed)**2).mean(dim=1).numpy()

    

    if threshold is None:

        threshold = np.mean(mse) + 2*np.std(mse)

    

    anomalies = mse > threshold

    return anomalies, mse, threshold





if __name__ == '__main__':

    # Example usage:

    np.random.seed(0)

    

    # Create sample �normal� log data with 20 features per sample

    normal_data = np.random.normal(0, 1, (1000, 20))

    # Simulate anomalies by drawing from a different distribution

    anomalous_data = np.random.normal(0, 5, (50, 20))

    data = np.vstack([normal_data, anomalous_data])

    

    # Train the autoencoder on the data

    model = train_autoencoder(data, input_size=20, latent_dim=8, epochs=100)

    

    # Run anomaly detection

    anomalies, mse_scores, thresh = detect_anomalies(model, data)

    print("Anomaly threshold:", thresh)

    print("Number of anomalies detected:", np.sum(anomalies))

