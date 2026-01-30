import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

from dataset import StrokeDataset
from model import StrokeLSTM

# ---------------- CONFIG ----------------
DATA_DIR = "../data/raw"
MODEL_PATH = "stroke_lstm.pth"
OUTPUT_DIR = "evaluation_outputs"
MAX_SEQ_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- BÉZIER SMOOTHING ----------------
def bezier_smooth(points, samples_per_segment=8):
    """
    Applies piecewise cubic Bézier smoothing to a stroke.

    Args:
        points (np.ndarray): Shape (N, 2), raw stroke points
        samples_per_segment (int): Points sampled per Bézier segment

    Returns:
        np.ndarray: Smoothed stroke with same length as input
    """
    n = len(points)

    # Too short to smooth
    if n < 4:
        return points.copy()

    smoothed = []

    # Process stroke in overlapping windows
    for i in range(0, n - 3, 3):
        p0 = points[i]
        p1 = points[i + 1]
        p2 = points[i + 2]
        p3 = points[i + 3]

        t_vals = np.linspace(0, 1, samples_per_segment, endpoint=False)

        for t in t_vals:
            b = (
                (1 - t) ** 3 * p0 +
                3 * (1 - t) ** 2 * t * p1 +
                3 * (1 - t) * t ** 2 * p2 +
                t ** 3 * p3
            )
            smoothed.append(b)

    smoothed.append(points[-1])

    smoothed = np.array(smoothed)

    # Resample to match original stroke length
    if len(smoothed) != n:
        indices = np.linspace(0, len(smoothed) - 1, n)
        resampled = np.zeros((n, 2))

        for d in range(2):
            resampled[:, d] = np.interp(
                indices,
                np.arange(len(smoothed)),
                smoothed[:, d]
            )
        return resampled

    return smoothed


# ---------------- ERROR METRICS ----------------
def compute_metrics(pred, gt):
    mse = np.mean((pred - gt) ** 2)
    mae = np.mean(np.abs(pred - gt))
    return mse, mae

# ---------------- LOAD MODEL ----------------
model = StrokeLSTM()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

criterion = nn.MSELoss()

# ---------------- LOAD DATASET ----------------
dataset = StrokeDataset(
    data_dir=DATA_DIR,
    max_seq_len=MAX_SEQ_LEN,
    noise_level=0.02
)

results = []

# ---------------- EVALUATION LOOP ----------------
for idx in range(len(dataset)):
    noisy, clean = dataset[idx]

    noisy = noisy.unsqueeze(0).to(DEVICE)
    clean_np = clean.numpy()

    # ML prediction
    with torch.no_grad():
        pred_ml = model(noisy).squeeze(0).cpu().numpy()

    # Raw stroke (noisy x,y only)
    raw_xy = noisy.squeeze(0).cpu().numpy()[:, :2]

    # Bézier smoothing
    bezier_xy = bezier_smooth(raw_xy)

    # Metrics
    mse_raw, mae_raw = compute_metrics(raw_xy, clean_np)
    mse_bez, mae_bez = compute_metrics(bezier_xy, clean_np)
    mse_ml, mae_ml = compute_metrics(pred_ml, clean_np)

    results.append({
        "stroke_id": idx,
        "mse_raw": mse_raw,
        "mae_raw": mae_raw,
        "mse_bezier": mse_bez,
        "mae_bezier": mae_bez,
        "mse_ml": mse_ml,
        "mae_ml": mae_ml
    })

    # ---------------- PLOT ----------------
    plt.figure(figsize=(6, 6))
    plt.plot(raw_xy[:, 0], raw_xy[:, 1], "k--", label="Raw")
    plt.plot(bezier_xy[:, 0], bezier_xy[:, 1], "b", label="Bezier")
    plt.plot(pred_ml[:, 0], pred_ml[:, 1], "g", label="ML")
    plt.plot(clean_np[:, 0], clean_np[:, 1], "r", label="Ground Truth")

    plt.legend()
    plt.title(f"Stroke {idx}")
    plt.axis("equal")
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, f"stroke_{idx:03d}.png"))
    plt.close()

# ---------------- SAVE METRICS ----------------
df = pd.DataFrame(results)
df.to_csv(os.path.join(OUTPUT_DIR, "evaluation_metrics.csv"), index=False)

print("Evaluation complete.")
print(df.mean())
