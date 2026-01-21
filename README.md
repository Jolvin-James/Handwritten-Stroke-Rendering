# Handwritten Stroke Recognition

A lightweight, web-based drawing application capable of capturing detailed stroke data for handwriting recognition tasks. This project serves as the frontend data collection interface, capturing high-fidelity stylus usage including pressure sensitivity and timing.

## Features

-   **High-Performance Drawing Board**: Built on the HTML5 Canvas API.
-   **Advanced Input Support**: Uses Pointer Events API to support:
    -   Mouse drawing
    -   Touch drawing (Mobile/Tablet)
    -   Stylus/Pen with **Pressure Sensitivity**
-   **Data Capture**: Records standardized stroke data for AI/ML processing:
    -   `x`, `y` coordinates
    -   `time` (timestamp for velocity/order)
    -   `pressure` (force applied)
-   **Customizable Tools**:
    -   Adjustable Brush Size
    -   Color Picker
    -   Clear Canvas

## Tech Stack

-   **HTML5**
-   **CSS3** (Flexbox/Grid)
-   **Vanilla JavaScript** (No external libraries required)

## Setup & Usage

1.  Clone the repository or download the files.
2.  Open `frontend/index.html` in any modern web browser.
3.  Start drawing!
    -   Use a stylus on a compatible device to see pressure sensitivity in action.
    -   Open the browser's Developer Console (`F12`) to see logged stroke data arrays when you finish a stroke.

## Data Structure

The application captures data in the following format:

```javascript
[
  [ // Stroke 1
    { x: 100, y: 150, time: 17054812301, pressure: 0.5 },
    { x: 102, y: 152, time: 17054812316, pressure: 0.6 },
    ...
  ],
  [ // Stroke 2
    ...
  ]
]
```

## Machine Learning

The `ml/` directory contains Python scripts for processing the captured stroke data.

### Dataset Loader (`ml/dataset.py`)
The `StrokeDataset` class loads JSON stroke files and processes them into tensor sequences for training.

**Computed Features (5 dimensions):**
1.  **x_centered**: Bounding-box normalized X coordinate (centered).
2.  **y_centered**: Bounding-box normalized Y coordinate (centered).
3.  **dt_norm**: Normalized time delta (time since previous point).
4.  **p_norm**: Normalized pressure.
5.  **speed**: Calculated velocity magnitude (derived from dx/dy and dt).

### Usage

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the test script to verify data loading:
    ```bash
    cd ml
    python test_dataset.py
    ```
