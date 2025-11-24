import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
import os
from transformer_model import Transformer


class SyntheticDataset(Dataset):
    def __init__(self, data_file, settings_file):
        # Load data and settings
        self.data_df = pd.read_csv(data_file)
        with open(settings_file, 'r') as f:
            self.settings = json.load(f)

        # Identify continuous and discrete variables
        self.cont_vars = [
            var for var in self.settings["variable_distributions"]
            if self.settings["variable_distributions"][var]["type"] in ["continuous", "normal", "uniform"]
               and var != "Z"
        ]
        self.disc_vars = [
            var for var in self.settings["variable_distributions"]
            if self.settings["variable_distributions"][var]["type"] == "discrete"
               and var != "Z"
        ]

        # Prepare continuous and discrete data
        self.x_cont = torch.tensor(
            self.data_df[self.cont_vars].values, dtype=torch.float32
        ) if self.cont_vars else torch.zeros((len(self.data_df), 0), dtype=torch.float32)

        # Map discrete levels to 0-based indices
        self.level_mappings = {
            var: {level: idx for idx, level in enumerate(sorted(set(self.data_df[var])))}
            for var in self.disc_vars
        }
        self.x_disc = torch.tensor([
            [self.level_mappings[var][val] for var, val in zip(self.disc_vars, row)]
            for row in self.data_df[self.disc_vars].values
        ], dtype=torch.long) if self.disc_vars else torch.zeros((len(self.data_df), 0), dtype=torch.long)

        self.y = torch.tensor(self.data_df["Y"].values, dtype=torch.long)
        self.discrete_dims = [
            len(self.settings["variable_distributions"][var]["levels"])
            for var in self.disc_vars
        ]

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        return self.x_cont[idx], self.x_disc[idx], self.y[idx]

    def get_dims(self):
        return len(self.cont_vars), self.discrete_dims, self.settings["n_classes"]


def train_model(data_file, settings_file, epochs=50, batch_size=64, lr=0.001,
                device="cuda" if torch.cuda.is_available() else "cpu"):
    # Initialize dataset
    dataset = SyntheticDataset(data_file, settings_file)
    cont_dim, discrete_dims, num_classes = dataset.get_dims()

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    model = Transformer(cont_dim, discrete_dims, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6,
                                                     verbose=True)
    # Optional: Cosine annealing for initial warm-up
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)

    # Training loop
    best_val_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = f"checkpoints/model_{dataset.settings['dataset_id']}.pth"
    metrics = {"train_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x_cont, x_disc, y in train_loader:
            x_cont, x_disc, y = x_cont.to(device), x_disc.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x_cont, x_disc)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_cont.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x_cont, x_disc, y in val_loader:
                x_cont, x_disc, y = x_cont.to(device), x_disc.to(device), y.to(device)
                outputs = model(x_cont, x_disc)
                _, predicted = torch.max(outputs, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
                loss = criterion(outputs, y)
                val_loss += loss.item() * x_cont.size(0)

        val_acc = val_correct / val_total
        val_loss /= val_total
        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)
        metrics["val_acc"].append(val_acc)

        # Update learning rate
        if epoch < 20:  # Use cosine annealing for first 20 epochs
            scheduler_cosine.step()
        else:  # Switch to ReduceLROnPlateau
            scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)

    # Save metrics
    with open(f"checkpoints/metrics_{dataset.settings['dataset_id']}.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return best_val_acc


if __name__ == "__main__":
    dataset_dir = "datasets"
    for i in range(1, len(dataset_configs) + 1):
        data_file = f"{dataset_dir}/ds_{i:03d}_data.csv"
        settings_file = f"{dataset_dir}/ds_{i:03d}_settings.json"
        if os.path.exists(data_file) and os.path.exists(settings_file):
            print(f"Training on dataset ds_{i:03d}")
            best_val_acc = train_model(data_file, settings_file)
            print(f"Best validation accuracy for ds_{i:03d}: {best_val_acc:.4f}")
        else:
            print(f"Dataset ds_{i:03d} not found")