# Handwritten Stroke Recognition & Smoothing

## Overview
This project focuses on collecting and processing handwritten stroke data for machine learning applications, specifically for stroke smoothing and denoising. It includes a web-based frontend for capturing high-fidelity stroke data (coordinates, pressure, timestamp) and a Python/PyTorch backend for data loading and feature engineering.

## Features

### Frontend (Data Collection)
- **Web Interface**: A clean, responsive drawing board.
- **High-Fidelity Capture**: Records:
  - `x`, `y` coordinates (normalized 0-1)
  - `t` (timestamp in ms)
  - `p` (pressure sensitivity, supported on compatible devices/tablets)
- **JSON Export**: One-click export of drawn strokes to JSON format for dataset creation.
- **Customization**: Adjustable stroke color and line width.

### Machine Learning (Backend)
- **PyTorch Dataset**: Custom `StrokeDataset` class (`ml/dataset.py`) for efficient data loading.
- **Feature Engineering**:
  - Coordinate normalization (preserving aspect ratio).
  - Velocity and pressure normalization.
  - Derived features: Delta time (`dt`) and Speed.
- **Synthetic Noise**: Built-in augmentation to add Gaussian noise to strokes, enabling training of denoising models.
- **Stroke Smoothing Model**: An LSTM-based regression model (`ml/model.py`) designed to smooth and denoise handwritten strokes.

## Project Structure

```
.
├── frontend/           # Web application
│   ├── index.html      # Main entry point
│   ├── styles.css      # Styling
│   └── canvas.js       # Drawing logic & data capture
├── ml/                 # Machine Learning pipeline
│   ├── dataset.py      # PyTorch Dataset implementation
│   ├── model.py        # LSTM Model architecture
│   ├── train.py        # Training script
│   └── test_dataset.py # Verification script
├── data/               # Directory for storing stroke JSON files
└── requirements.txt    # Python dependencies
```

## Getting Started

### 1. Frontend Setup
1. Open `frontend/index.html` in any modern web browser.
2. Start drawing on the canvas.
3. Click **Export JSON** to download your session data.
4. Move the downloaded JSON files into the `data/` (or `data/raw/`) directory.

### 2. Python Environment
Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Using the Dataset
You can load and inspect the dataset using the provided scripts.

**Testing the Pipeline:**
Run the test script to verify data loading and feature shape:
```bash
python -m ml.test_dataset
```

**Using in Code:**
```python
from ml.dataset import StrokeDataset

# Load data from the 'data' directory
dataset = StrokeDataset(data_dir="data", max_seq_len=128)

# Get a sample (Returns: noisy_input, clean_target)
input_stroke, target_stroke = dataset[0]
print(input_stroke.shape)  # Expected: (128, 5) -> [x, y, dt, p, v]
```

### 4. Using the Model
You can instantiate the LSTM model for training or inference:

```python
from ml.model import StrokeLSTM
import torch

# Initialize model
model = StrokeLSTM(input_size=5, hidden_size=128, output_size=2)

# Create a dummy batch (batch_size=1, seq_len=128, features=5)
dummy_input = torch.randn(1, 128, 5)

# Forward pass
output = model(dummy_input)
print(output.shape) # Expected: (1, 128, 2) -> [x, y]
```

### 5. Training the Model
To train the LSTM model on your collected data:

1. Ensure your JSON data files are in `data/raw/` (or configure `DATA_DIR` in `ml/train.py`).
2. Run the training module:

```bash
python -m ml.train
```

**Training Details:**
- **Epochs**: 50
- **Batch Size**: 16
- **Device**: CUDA (if available) else CPU
- **Output**: The best model weights are saved to `stroke_lstm.pth`.


## Data Format
Exported JSON files contain a list of strokes, where each stroke is a list of points:
```json
{
  "strokes": [
    {
      "stroke_id": 1,
      "points": [
        { "x": 0.45, "y": 0.32, "t": 120.5, "p": 0.5 },
        ...
      ]
    }
  ]
}
```
