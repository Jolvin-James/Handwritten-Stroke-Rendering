# ml/evaluate.py
import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import time

from dataset import StrokeDataset
from model import StrokeLSTM

# ---------------- CONFIG ----------------
DATA_DIR = "../data/raw"
MODEL_PATH = "stroke_lstm.pth"
OUTPUT_DIR = "evaluation_outputs"
MAX_SEQ_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

inference_times = []

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- ADAPTIVE BÉZIER SMOOTHING ----------------
def bezier_smooth(points, min_samples=4, max_samples=12):
    """
    Applies velocity-adaptive cubic Bézier smoothing.

    Sampling density increases in high-motion regions to avoid over-smoothing
    and decreases in stable regions to avoid unnecessary regularization.

    Args:
        points (np.ndarray): Shape (N, 2)
        min_samples (int): Minimum samples per Bézier segment
        max_samples (int): Maximum samples per Bézier segment

    Returns:
        np.ndarray: Smoothed stroke with same length as input
    """
    n = len(points)

    if n < 4:
        return points.copy()

    # Compute velocity magnitude
    diffs = np.diff(points, axis=0)
    speed = np.linalg.norm(diffs, axis=1)
    speed = np.concatenate([[speed[0]], speed])

    # Normalize speed
    speed_norm = speed / (speed.max() + 1e-6)

    smoothed = []

    for i in range(0, n - 3, 3):
        p0 = points[i]
        p1 = points[i + 1]
        p2 = points[i + 2]
        p3 = points[i + 3]

        # Adaptive sampling based on local speed
        local_speed = np.mean(speed_norm[i:i + 4])
        samples = int(
            min_samples + (max_samples - min_samples) * local_speed
        )

        t_vals = np.linspace(0, 1, samples, endpoint=False)

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

    # Resample to original length
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


# ---------------- STROKE CONTINUITY ----------------
def compute_continuity(stroke_xy):
    """
    Measures stroke smoothness using point-to-point distance consistency.
    Lower value => smoother, more continuous stroke.

    Args:
        stroke_xy (np.ndarray): Shape (N, 2)

    Returns:
        float: continuity score (std of distances)
    """
    if len(stroke_xy) < 2:
        return 0.0

    diffs = np.diff(stroke_xy, axis=0)
    distances = np.linalg.norm(diffs, axis=1)

    return np.std(distances)


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

    # ML prediction with latency measurement
    start_time = time.time()
    with torch.no_grad():
        pred_ml = model(noisy).squeeze(0).cpu().numpy()
    inference_times.append(time.time() - start_time)

    # Raw stroke (noisy x,y only)
    raw_xy = noisy.squeeze(0).cpu().numpy()[:, :2]

    # Bézier smoothing
    bezier_xy = bezier_smooth(raw_xy)

    # Stroke continuity (lower is smoother)
    cont_raw = compute_continuity(raw_xy)
    cont_bez = compute_continuity(bezier_xy)
    cont_ml = compute_continuity(pred_ml)

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
        "mae_ml": mae_ml,
        "cont_raw": cont_raw,
        "cont_bezier": cont_bez,
        "cont_ml": cont_ml
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

# ---------------- AGGREGATE VISUALIZATIONS ----------------
# Average Error Comparison
avg_errors = df[[
    "mse_raw", "mse_bezier", "mse_ml"
]].mean()

plt.figure(figsize=(6, 4))
avg_errors.plot(kind="bar")
plt.ylabel("MSE")
plt.title("Average MSE Comparison")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "avg_mse_comparison.png"))
plt.close()

# Average MAE Comparison
avg_mae = df[[
    "mae_raw", "mae_bezier", "mae_ml"
]].mean()

plt.figure(figsize=(6, 4))
avg_mae.plot(kind="bar")
plt.ylabel("MAE")
plt.title("Average MAE Comparison")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "avg_mae_comparison.png"))
plt.close()

# Average Continuity Comparison
avg_cont = df[[
    "cont_raw", "cont_bezier", "cont_ml"
]].mean()

plt.figure(figsize=(6, 4))
avg_cont.plot(kind="bar")
plt.ylabel("Continuity Score (Lower is Smoother)")
plt.title("Average Stroke Continuity Comparison")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "avg_continuity_comparison.png"))
plt.close()

print("Evaluation complete.")
print(df.mean())

# ---------------- INFERENCE LATENCY ----------------
avg_latency = np.mean(inference_times)
std_latency = np.std(inference_times)

print(f"Average inference latency per stroke: {avg_latency * 1000:.3f} ms")
print(f"Latency standard deviation: {std_latency * 1000:.3f} ms")
