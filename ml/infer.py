# ml/infer.py
import torch
import numpy as np
from dataset import StrokeDataset

MODEL_PATH = "stroke_lstm_ts.pt"
DATA_DIR = "../data/raw"   
DEVICE = "cpu"
MAX_SEQ_LEN = 128

# Load Model
model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
model.eval()

# Load Preprocessor
_preprocessor = StrokeDataset(
    data_dir=DATA_DIR,
    max_seq_len=MAX_SEQ_LEN,
    noise_level=0.0  # not used during inference
)

# Inference
def smooth_stroke(points):
    """
    Args:
        points (list[dict]): raw stroke points from canvas
                             each point has x, y, t, p

    Returns:
        list[list]: smoothed (x, y) points, length = MAX_SEQ_LEN
    """

    if len(points) < 3:
        return [[p["x"], p["y"]] for p in points]

    features, meta = _preprocessor._process_stroke(points, return_meta=True)

    if features is None:
        return [[p["x"], p["y"]] for p in points]

    # Shape: (1, seq_len, 5)
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred = model(x).squeeze(0).cpu().numpy()  # (seq_len, 2)

    return {
        "points": pred.tolist(),
        "meta": meta
    }
