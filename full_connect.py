import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# config
DATA_PATH = './data/data.csv'
MODEL_PATH = './model/fc_model.pth'
TRAIN_SIZE = 1500

# load data
df = pd.read_csv(DATA_PATH)

def parse_array(col):
    return np.array(eval(col))  # transfer the string to np

r = np.stack(df['r'].apply(parse_array))
alpha = np.stack(df['alpha'].apply(parse_array))
beta = np.stack(df['beta'].apply(parse_array))
x = np.stack(df['x'].apply(parse_array))

# dataset definition
class PortfolioDataset(Dataset):
    def __init__(self, r, alpha, beta, x):
        self.conditions = torch.tensor(np.hstack([r, alpha, beta]), dtype=torch.float32)
        self.targets = torch.tensor(x, dtype=torch.float32)

    def __len__(self):
        return len(self.conditions)

    def __getitem__(self, idx):
        return self.conditions[idx], self.targets[idx]

# dataloader
train_dataset = PortfolioDataset(r[:TRAIN_SIZE], alpha[:TRAIN_SIZE], beta[:TRAIN_SIZE], x[:TRAIN_SIZE])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# model - fully connect
class FCModel(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim=128):
        super(FCModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, input_dim)
        )

    def forward(self, condition):
        return self.model(condition)

# init model
input_dim = train_dataset.targets.shape[1]
condition_dim = train_dataset.conditions.shape[1]
model = FCModel(input_dim=input_dim, condition_dim=condition_dim)
optimizer = optim.Adam(model.parameters(), lr=5e-3)
loss_fn = nn.MSELoss()

# training
num_epochs = 1000
for epoch in range(num_epochs):
    total_loss = 0
    for condition, target in train_loader:
        output = model(condition)
        # print(output.shape)
        # print(target.shape)
        loss = loss_fn(output, target)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

# save the trained model
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
