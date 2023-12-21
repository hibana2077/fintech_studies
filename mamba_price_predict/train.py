import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import argparse
from mamba_ssm import Mamba
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Set up argument parser

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--load_data_location', type=str, default='data/processed_data.csv')
parser.add_argument('--save_model_location', type=str, default='model.pth')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--print_interval', type=int, default=10)
parser.add_argument('--window_size', type=int, default=10)

args = parser.parse_args()

# Set up constants

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = args.epochs
print_interval = args.print_interval
window_size = args.window_size

# Load data

data = pd.read_csv(args.load_data_location)
data.set_index('time', inplace=True)
data.sort_index(inplace=True)
data = data.astype(float)

# Split data

def split_data(data:pd.DataFrame, window_size:int):
    data.drop('time', axis=1, inplace=True)
    X = []
    y = []
    for i in range(len(data) - window_size):
        X.append(data.iloc[i:i+window_size].values)
        y.append(data.iloc[i+window_size].values)
    return pd.DataFrame(X), pd.DataFrame(y)

X, y = split_data(data, window_size)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Convert data to tensors

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

# Create dataloaders

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Create an instance of your model
batch, length, dim = args.batch_size, window_size, X_train.shape[2]
model = Mamba(
    d_model=dim,
    d_state=dim,
    d_conv=4,
    expand=2
).to(DEVICE)

# Define your loss function
criterion = nn.CrossEntropyLoss().to(DEVICE)

# Define your optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(EPOCHS):
    # Set the model to training mode
    model.train()

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training progress
        if (batch_idx + 1) % print_interval == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'model.pth')