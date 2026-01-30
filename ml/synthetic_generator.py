# ml/synthetic_generator.py
import os
import json
import numpy as np

# CONFIG
OUTPUT_DIR = "../data/raw"
NUM_FILES = 1200
STROKES_PER_FILE = 1
MIN_POINTS = 80
MAX_POINTS = 220

np.random.seed(42 + os.getpid())

# UTILITY FUNCTIONS
def generate_time(x, y):
    curvature = compute_curvature(x, y)

    # Normalize curvature
    curvature = curvature / (curvature.max() + 1e-6)

    # Slower where curvature is high
    base_dt = np.random.uniform(5, 10, size=len(x))
    dt = base_dt * (1 + 2.0 * curvature)

    t = np.cumsum(dt)
    return t - t[0]

def generate_pressure(x, y):
    curvature = compute_curvature(x, y)
    curvature = curvature / (curvature.max() + 1e-6)

    # Base pressure
    base_p = np.random.normal(0.7, 0.03, size=len(x))

    # Increase pressure at high curvature
    p = base_p + 0.15 * curvature

    return np.clip(p, 0.3, 1.0)

def add_spatial_jitter(x, y, scale):
    x_jitter = x + np.random.normal(0, scale, size=len(x))
    y_jitter = y + np.random.normal(0, scale, size=len(y))

    x_jitter = np.clip(x_jitter, 0.0, 1.0)
    y_jitter = np.clip(y_jitter, 0.0, 1.0)

    return x_jitter, y_jitter

def compute_curvature(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-6)
    return curvature

# STROKE GENERATORS
def line_stroke(n):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n) * np.random.uniform(0.3, 0.9)
    return x, y

def circle_stroke(n):
    theta = np.linspace(0, 2 * np.pi, n)
    r = 0.4
    x = 0.5 + r * np.cos(theta)
    y = 0.5 + r * np.sin(theta)
    return x, y

def sine_stroke(n):
    x = np.linspace(0, 1, n)
    y = 0.5 + 0.2 * np.sin(2 * np.pi * x * np.random.uniform(1, 3))
    return x, y

def spiral_stroke(n):
    theta = np.linspace(0, 4 * np.pi, n)
    r = np.linspace(0.05, 0.45, n)
    x = 0.5 + r * np.cos(theta)
    y = 0.5 + r * np.sin(theta)
    return x, y

def bezier_like_stroke(n):
    p0 = np.random.rand(2)
    p1 = np.random.rand(2)
    p2 = np.random.rand(2)

    t = np.linspace(0, 1, n)
    x = (1 - t)**2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
    y = (1 - t)**2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
    return x, y

STROKE_FUNCTIONS = [
    line_stroke,
    circle_stroke,
    sine_stroke,
    spiral_stroke,
    bezier_like_stroke
]

# STROKE CREATION
def create_stroke(stroke_id):
    n_points = np.random.randint(MIN_POINTS, MAX_POINTS)

    stroke_fn = np.random.choice(STROKE_FUNCTIONS)
    x, y = stroke_fn(n_points)

    scale = np.random.uniform(0.001, 0.003)
    x, y = add_spatial_jitter(x, y, scale=scale)

    t = generate_time(x, y)
    p = generate_pressure(x, y)

    points = []
    for i in range(n_points):
        points.append({
            "x": float(x[i]),
            "y": float(y[i]),
            "t": float(t[i]),
            "p": float(p[i])
        })

    return {
        "stroke_id": stroke_id,
        "points": points
    }

# FILE GENERATION
def generate_files():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for file_idx in range(NUM_FILES):
        strokes = []
        for s in range(STROKES_PER_FILE):
            strokes.append(create_stroke(s + 1))

        data = {
            "canvas": {
                "width": 1,
                "height": 1
            },
            "strokes": strokes
        }

        filename = f"strokes_{str(file_idx + 1).zfill(3)}.json"
        path = os.path.join(OUTPUT_DIR, filename)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved {path}")

if __name__ == "__main__":
    generate_files()
