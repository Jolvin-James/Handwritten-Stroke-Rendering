# ml/export_model.py
import torch
from model import StrokeLSTM

MODEL_WEIGHTS = "stroke_lstm.pth"
EXPORT_PATH = "stroke_lstm_ts.pt"
DEVICE = "cpu"
SEQ_LEN = 128
INPUT_SIZE = 5

# load model
model = StrokeLSTM(input_size=INPUT_SIZE, output_size=2)
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# dummy input
# Shape: (batch_size=1, seq_len, input_size)
dummy_input = torch.zeros(1, SEQ_LEN, INPUT_SIZE)

# export
with torch.no_grad():
    scripted_model = torch.jit.trace(model, dummy_input)

scripted_model.save(EXPORT_PATH)

print(f"TorchScript model exported to {EXPORT_PATH}")
