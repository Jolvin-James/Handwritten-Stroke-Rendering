# ml/train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import StrokeDataset
from model import StrokeLSTM

# Config
DATA_DIR = "../data/raw"
BATCH_SIZE = 12
EPOCHS = 40
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "stroke_lstm.pth"

# Dataset & Loader
dataset = StrokeDataset(
    data_dir=DATA_DIR,
    max_seq_len=128,
    noise_level=0.02
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Model, Loss, Optimizer
model = StrokeLSTM().to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
print(f"Training on {DEVICE}")
print(f"Total samples: {len(dataset)}")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for noisy, clean in loader:
        noisy = noisy.to(DEVICE)     # (B, T, 5)
        clean = clean.to(DEVICE)     # (B, T, 2)

        optimizer.zero_grad()
        preds = model(noisy)
        loss = criterion(preds, clean)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {avg_loss:.6f}")

# Save Model Weights
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model weights saved to {MODEL_PATH}")
