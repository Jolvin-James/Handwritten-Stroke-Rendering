# Handwritten Stroke Recognition & Smoothing

> A full-stack machine learning application for capturing, processing, and smoothing handwritten strokes in real-time using LSTM neural networks.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)

---

## Overview

This project demonstrates end-to-end machine learning for handwriting analysis, combining:
- **High-fidelity data capture** via HTML5 Canvas with pressure sensitivity
- **Advanced feature engineering** with temporal and spatial features
- **LSTM-based neural network** for stroke denoising and smoothing
- **Real-time inference** through a Flask API with TorchScript optimization

Perfect for learning about sequence modeling, data preprocessing, and deploying ML models in production.

---

## Features

### Frontend (Interactive Canvas)

- **Responsive Drawing Board**: Full-screen HTML5 canvas with smooth rendering
- **Multi-Modal Input**: Supports mouse, touch, and stylus (with pressure sensitivity)
- **High-Fidelity Capture**: Records 4D stroke data:
  - `x`, `y` — Normalized coordinates (0-1 range, resolution-independent)
  - `t` — Relative timestamp in milliseconds
  - `p` — Pressure sensitivity (0-1, defaults to 1.0 for non-stylus devices)
- **Real-Time ML Smoothing**: Instant visual feedback with smoothed strokes overlaid
- **Data Export**: One-click JSON export with auto-incrementing filenames
- **Customization**: Adjustable stroke color and line width

### Machine Learning Pipeline

#### Model Architecture
- **LSTM-Based Regression**: 2-layer bidirectional LSTM with 128 hidden units
- **Input Features** (5D): `[x, y, Δt, pressure, speed]`
- **Output**: Smoothed `(x, y)` coordinates for each timestep
- **Regularization**: Dropout (0.2) to prevent overfitting

#### Data Processing
- **Synthetic Data Generator**: Creates ~1200 diverse training samples
  - Geometric shapes: lines, circles, spirals, Bézier curves
  - Human-like variations: speed modulation, pressure simulation
- **Aspect-Ratio Preserving Normalization**: Prevents shape distortion
- **Sequence Resampling**: Linear interpolation to fixed length (128 points)
- **On-the-Fly Augmentation**:
  - Gaussian noise injection
  - Random rotation (±15°)
  - Random scaling (±20%)

#### Inference Optimization
- **TorchScript Export**: Serialized model for faster inference
- **Flask REST API**: Lightweight server with CORS support
- **Low Latency**: Pre-loaded model with `torch.no_grad()` optimization

---

## Project Structure

```
Handwritten-Stroke-Recognition/
│
├── frontend/                    # Web application
│   ├── index.html               # Main UI entry point
│   ├── styles.css               # Responsive styling with flexbox
│   └── canvas.js                # Canvas logic, event handling, API calls
│
├── ml/                          # Machine Learning pipeline
│   ├── app.py                   # Flask API server (CORS-enabled)
│   ├── dataset.py               # PyTorch Dataset with augmentation
│   ├── model.py                 # LSTM architecture definition
│   ├── train.py                 # Training loop with validation
│   ├── infer.py                 # TorchScript inference wrapper
│   ├── export_model.py          # Model export to TorchScript
│   ├── synthetic_generator.py  # Synthetic data creation
│   ├── evaluate.py              # Model evaluation metrics
│   ├── test_dataset.py          # Dataset verification script
│   ├── stroke_lstm.pth          # Trained model weights
│   └── stroke_lstm_ts.pt        # TorchScript optimized model
│
├── data/                        # Stroke data storage (JSON files)
│   └── raw/                     # Raw collected/generated strokes
│
├── requirements.txt             # Python dependencies
├── PREP.md                      # Interview preparation guide
└── README.md                    # This file
```

---

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Modern web browser (Chrome, Firefox, Edge, Safari)
- (Optional) Stylus/tablet for pressure sensitivity

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Jolvin-James/Handwritten-Stroke-Rendering.git
   cd Handwritten-Stroke-Recognition
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

#### Option 1: Full Stack (Recommended)

1. **Start the Flask backend**
   ```bash
   python -m ml.app
   ```
   Server starts at `http://localhost:5000`

2. **Open the frontend**
   - Double-click `frontend/index.html`, or
   - Serve with Python: `python -m http.server -d frontend 8080`
   - Navigate to `http://localhost:8080`

3. **Draw and watch the magic!** 
   - Draw strokes on the canvas
   - Release to see real-time ML smoothing
   - Export your data with the "Export" button

#### Option 2: Data Collection Only

Open `frontend/index.html` directly to collect stroke data without ML inference (backend not required).

---

## Training Your Own Model

### 1. Generate Synthetic Training Data

```bash
python -m ml.synthetic_generator
```

This creates ~1200 JSON files in `data/raw/` with diverse stroke patterns.

### 2. Verify Dataset

```bash
python -m ml.test_dataset
```

Expected output:
```
Loaded 1200 strokes
Sample input shape: torch.Size([128, 5])
Sample target shape: torch.Size([128, 2])
```

### 3. Train the Model

```bash
python -m ml.train
```

**Training Configuration:**
- **Epochs**: 40
- **Batch Size**: 12
- **Optimizer**: Adam (lr=1e-3)
- **Loss**: MSE (Mean Squared Error)
- **Device**: Auto-detects CUDA/CPU

**Output**: `stroke_lstm.pth` (best model weights)

### 4. Export for Production

```bash
python -m ml.export_model
```

Generates `stroke_lstm_ts.pt` (TorchScript format) for optimized inference.

### 5. Evaluate Performance

```bash
python -m ml.evaluate
```

Generates visualizations comparing original vs. smoothed strokes in `ml/evaluation_outputs/`.

---

## Data Format

### Input JSON Structure

```json
{
  "canvas": {
    "width": 1920,
    "height": 1080
  },
  "strokes": [
    {
      "stroke_id": 1,
      "points": [
        { "x": 0.45, "y": 0.32, "t": 0.0, "p": 0.8 },
        { "x": 0.46, "y": 0.33, "t": 16.7, "p": 0.85 },
        { "x": 0.47, "y": 0.34, "t": 33.4, "p": 0.9 }
      ]
    }
  ]
}
```

### Feature Engineering Pipeline

| Raw Data | Derived Features | Model Input |
|----------|------------------|-------------|
| `x, y` (normalized) | `Δt` (time delta) | 5D tensor: |
| `t` (timestamp) | `speed` (velocity magnitude) | `[x, y, Δt, p, speed]` |
| `p` (pressure) | Aspect-ratio normalization | Shape: `(128, 5)` |

---

## Use Cases

- **Handwriting Recognition**: Pre-processing step for OCR systems
- **Digital Art**: Stroke stabilization for drawing applications
- **Biometric Authentication**: Signature verification systems
- **Educational Tools**: Handwriting analysis and improvement
- **Research**: Sequence modeling, time-series prediction

---

## Technical Deep Dive

### Model Architecture

```python
StrokeLSTM(
  (lstm): LSTM(input_size=5, hidden_size=128, num_layers=2, dropout=0.2)
  (fc): Linear(in_features=128, out_features=2)
)
```

**Forward Pass:**
1. Input: `(batch, 128, 5)` — Noisy stroke features
2. LSTM: Captures temporal dependencies
3. FC Layer: Maps hidden states → `(x, y)` coordinates
4. Output: `(batch, 128, 2)` — Smoothed trajectory

### API Endpoint

**POST** `/smooth-stroke`

**Request:**
```json
{
  "points": [
    { "x": 0.5, "y": 0.5, "t": 0, "p": 1.0 },
    ...
  ]
}
```

**Response:**
```json
{
  "points": [[0.501, 0.499], [0.502, 0.500], ...],
  "meta": {
    "min_x": 0.45,
    "min_y": 0.30,
    "scale": 0.25,
    "center_x": 0.0,
    "center_y": 0.0
  }
}
```

---

## Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add Transformer-based architecture comparison
- [ ] Implement WebAssembly inference (remove backend dependency)
- [ ] Multi-stroke character recognition
- [ ] Real-time stroke classification (letters, shapes)
- [ ] Mobile app version (React Native/Flutter)

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## Contact

**Jolvin James**  
GitHub: [@Jolvin-James](https://github.com/Jolvin-James)  
Repository: [Handwritten-Stroke-Rendering](https://github.com/Jolvin-James/Handwritten-Stroke-Rendering)
