import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def preprocess_for_autoencoder(data):
    missing_mask = np.isnan(data)
    data_imputed = np.where(missing_mask, np.nanmean(data, axis=0), data)
    data_mean = np.mean(data_imputed, axis=0)
    data_std = np.std(data_imputed, axis=0)
    data_normalized = (data_imputed - data_mean) / data_std
    return torch.tensor(data_normalized, dtype=torch.float32), data_mean, data_std, missing_mask


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


def train_autoencoder(model, data_tensor, epochs=50, batch_size=256, learning_rate=0.001):
    dataset = TensorDataset(data_tensor, data_tensor)  # Input and target are the same
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")
    
    return model


def impute_missing_values_encoder(model, data_tensor, data_mean, data_std, missing_mask):
    model.eval()
    with torch.no_grad():
        reconstructed_data = model(data_tensor).numpy()
    reconstructed_data = reconstructed_data * data_std + data_mean
    imputed_data = data_tensor.numpy() * data_std + data_mean 
    imputed_data[missing_mask] = reconstructed_data[missing_mask] 
    return imputed_data


def impute_data(data):
    # Step 1: Preprocess the data
    data_tensor, data_mean, data_std, missing_mask = preprocess_for_autoencoder(data)

    # Step 2: Initialize the autoencoder
    input_dim = data_tensor.shape[1]
    autoencoder = Autoencoder(input_dim)

    # Step 3: Train the autoencoder
    autoencoder = train_autoencoder(autoencoder, data_tensor, epochs=50, batch_size=256, learning_rate=0.001)

    # Step 4: Impute missing values
    imputed_data = impute_missing_values_encoder(autoencoder, data_tensor, data_mean, data_std, missing_mask)
    return imputed_data