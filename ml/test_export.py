# ml/test_export.py
import torch

MODEL_PATH = "stroke_lstm_ts.pt"

model = torch.jit.load(MODEL_PATH)
model.eval()

# fake stroke input
x = torch.randn(1, 128, 5)

with torch.no_grad():
    y = model(x)

print("Output shape:", y.shape)
