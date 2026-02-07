# Data Directory

This directory contains all stroke data used for training, testing, and evaluating the Handwritten Stroke Recognition model.

---

## Directory Structure

```
data/
├── raw/                    # Synthetically generated training data (~1200 files)
├── drawn_strokes/          # Manually captured strokes from web frontend (~136 files)
└── processed/              # Reserved for cached preprocessed tensors (optional)
```

### Directory Descriptions

| Directory | Purpose | File Count | Source |
|-----------|---------|------------|--------|
| **raw/** | Primary training dataset with synthetic strokes | ~1200 | `ml/synthetic_generator.py` |
| **drawn_strokes/** | Real-world handwriting samples for testing | ~136 | `frontend/index.html` |
| **processed/** | Cached preprocessed data (optional) | Variable | Data pipeline |

---

## Data Format Specification

All stroke data is stored in **JSON** format with the following schema:

### JSON Schema

```json
{
  "canvas": {
    "width": 1,
    "height": 1
  },
  "strokes": [
    {
      "stroke_id": 1,
      "points": [
        {
          "x": 0.5,
          "y": 0.5,
          "t": 0.0,
          "p": 0.8
        },
        {
          "x": 0.51,
          "y": 0.52,
          "t": 16.7,
          "p": 0.85
        }
      ]
    }
  ]
}
```

### Field Descriptions

#### Canvas Object
- **width** (number): Canvas width in normalized units (typically 1)
- **height** (number): Canvas height in normalized units (typically 1)

#### Stroke Object
- **stroke_id** (integer): Unique identifier for the stroke within the file
- **points** (array): Ordered sequence of point objects representing the stroke trajectory

#### Point Object

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| **x** | float | [0.0, 1.0] | Normalized X coordinate (resolution-independent) |
| **y** | float | [0.0, 1.0+] | Normalized Y coordinate (aspect-ratio preserved) |
| **t** | float | [0.0, ∞) | Timestamp in milliseconds, relative to stroke start |
| **p** | float | [0.0, 1.0] | Pressure sensitivity (0.5 default for non-stylus) |

### Important Notes

1. **Coordinate Normalization**: 
   - Coordinates are normalized to [0, 1] range to ensure resolution independence
   - The `y` coordinate may slightly exceed 1.0 due to aspect-ratio preservation
   - This prevents shape distortion when scaling strokes

2. **Relative Timestamps**:
   - Time `t` is measured from the start of each stroke (first point has `t=0`)
   - Enables calculation of velocity and acceleration features
   - Captures the temporal dynamics of handwriting

3. **Pressure Simulation**:
   - Real data: Captured from stylus/tablet devices
   - Synthetic data: Simulated based on curve curvature (higher pressure on turns)
   - Defaults to 0.5 for mouse/touch input without pressure support

---

## Dataset Statistics

### Synthetic Data (raw/)

- **Total Files**: ~1200 JSON files
- **Stroke Types**: Lines, circles, spirals, sine waves, Bézier curves
- **Points per Stroke**: Variable (typically 50-200 points)
- **Features**: Simulated human-like variations in speed and pressure
- **Purpose**: Training the LSTM denoising model

### Real Data (drawn_strokes/)

- **Total Files**: ~136 JSON files
- **Source**: Manual drawings via web frontend
- **Characteristics**: Real human handwriting patterns
- **Purpose**: Testing and validation on real-world data

---

## Data Generation

### Generate Synthetic Training Data

```bash
python -m ml.synthetic_generator
```

**Output**: Creates ~1200 JSON files in `data/raw/` with diverse geometric patterns.

**Generated Shapes**:
- Straight lines (horizontal, vertical, diagonal)
- Circles (various radii)
- Spirals (Archimedean and logarithmic)
- Sine waves (different frequencies and amplitudes)
- Bézier curves (quadratic and cubic)

### Capture Real Handwriting Data

1. Open `frontend/index.html` in a web browser
2. Draw strokes on the canvas
3. Click "Export" to download JSON files
4. Save files to `data/drawn_strokes/` directory

---

## Data Usage

### Loading Data in Python

```python
import json
from pathlib import Path

# Load a single stroke file
def load_stroke_file(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

# Load all strokes from a directory
def load_all_strokes(directory):
    stroke_files = Path(directory).glob('*.json')
    all_strokes = []
    
    for file in stroke_files:
        data = load_stroke_file(file)
        all_strokes.extend(data['strokes'])
    
    return all_strokes

# Example usage
raw_strokes = load_all_strokes('data/raw')
print(f"Loaded {len(raw_strokes)} strokes")
```

### Using with PyTorch Dataset

```python
from ml.dataset import StrokeDataset

# Initialize dataset
dataset = StrokeDataset(
    data_dir="data/raw",
    max_seq_len=128
)

# Get a training sample
noisy_input, clean_target = dataset[0]

print(f"Input shape: {noisy_input.shape}")   # (128, 5) - [x, y, dt, p, speed]
print(f"Target shape: {clean_target.shape}") # (128, 2) - [x, y]
```

---

## Feature Engineering Pipeline

The raw 4D point data `(x, y, t, p)` is transformed into 5D features for model input:

| Raw Feature | Derived Feature | Description |
|-------------|-----------------|-------------|
| `x, y` | `x, y` (centered) | Coordinates normalized and centered at origin |
| `t` | `Δt` (normalized) | Time delta between consecutive points |
| `p` | `p` (normalized) | Standardized pressure values |
| `x, y, t` | `speed` | Velocity magnitude: √(Δx² + Δy²) / Δt |

**Final Model Input**: `[x, y, Δt, p, speed]` — Shape: `(128, 5)`

---

## Data Quality Guidelines

When capturing or generating stroke data, ensure:

1. **Sufficient Points**: Each stroke should have at least 10-15 points for meaningful features
2. **Smooth Sampling**: Points should be relatively evenly distributed along the stroke
3. **Realistic Timing**: Time intervals should reflect natural drawing speed
4. **Valid Pressure**: Pressure values should be in [0, 1] range
5. **Normalized Coordinates**: All coordinates should be properly normalized

---

## File Naming Convention

- **Synthetic Data**: `strokes_001.json`, `strokes_002.json`, etc.
- **Manual Data**: `strokes_001.json`, `strokes_002.json`, etc. (auto-incremented by frontend)

Files are numbered sequentially with zero-padding for proper sorting.

---

## Troubleshooting

### Common Issues

**Issue**: Dataset loader fails to find files
- **Solution**: Ensure JSON files are directly in `data/raw/` or `data/drawn_strokes/`, not in subdirectories

**Issue**: Invalid JSON format errors
- **Solution**: Validate JSON structure matches the schema above. Check for missing fields or incorrect data types

**Issue**: Strokes have too few points
- **Solution**: Adjust the synthetic generator parameters or draw longer strokes in the frontend

**Issue**: Memory errors when loading large datasets
- **Solution**: Use the PyTorch DataLoader with batching instead of loading all data at once

---

## Additional Resources

- **Main README**: [../README.md](../README.md) - Project overview and setup
- **Dataset Implementation**: [../ml/dataset.py](../ml/dataset.py) - PyTorch Dataset class
- **Synthetic Generator**: [../ml/synthetic_generator.py](../ml/synthetic_generator.py) - Data generation script
- **Frontend Capture**: [../frontend/canvas.js](../frontend/canvas.js) - Data capture logic
