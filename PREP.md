# Interview Preparation Guide: Handwritten Stroke Recognition

This guide covers potential interview questions and concepts based on your current codebase. Since your project captures high-fidelity stroke data and trains a **Machine Learning Model** to smooth it, questions will focus on **HTML5 Canvas**, **Data Feature Engineering**, and **LSTM/RNN Training**.

This guide is updated to reflect the full stack implementation, including the backend training pipeline.

---

## Part 1: The Existing Code (Frontend Engineering)

These questions directly relate to the `frontend/index.html`, `frontend/styles.css`, and `frontend/canvas.js` files you have written.

### 1. HTML5 Canvas API
**Context**: You rely entirely on the `<canvas>` element for this project.

*   **Q: What is the HTML5 Canvas?**
    *   *Answer*: It is an HTML element used to draw graphics via scripting (usually JavaScript). It's pixel-based (raster), meaning once you draw something, it becomes part of the canvas bitmap (unlike SVG which is vector-based).

*   **Q: Explain `getContext('2d')`.**
    *   *Answer*: This method returns a "drawing context" on the canvas. It provides the methods and properties needed to draw text, lines, boxes, etc. (e.g., `ctx.stroke()`, `ctx.lineWidth`).

*   **Q: The user draws a continuous line. How is this achieved code-wise?**
    *   *Reference*: `draw` function in `frontend/canvas.js`.
    *   *Answer*: It relies on `pointermove` events firing rapidly.
        1.  **`beginPath()`**: Starts a new path (prevents connecting to previous lines).
        2.  **`lineWidth` & `lineCap`**: Sets style. `round` makes the line edges smooth.
        3.  **`lineTo(x, y)`**: Defines a sub-path to the new mouse/pointer coordinates.
        4.  **`stroke()`**: Actually draws the line defined by `lineTo`.
        *Note*: Your code now correctly calls `beginPath` on `pointerdown` to ensure every new stroke is independent.

### 2. JavaScript Event Handling
**Context**: You use `pointerdown`, `pointerup`, `pointermove`, and `pointerleave`.

*   **Q: Why do we need `e.clientX - canvasOffsetX`?**
    *   *Reference*: Inside the `draw` function and event listeners in `frontend/canvas.js`.
    *   *Answer*: `e.clientX` gives the mouse/pointer position relative to the *browser viewport*. The Canvas starts at a specific position on the page (`offsetLeft`). To get the X coordinate *inside* the canvas, we must subtract the canvas's starting position from the pointer's position.

*   **Q: What is the purpose of the `isPainting` variable?**
    *   *Answer*: It acts as a "flag" or state variable. The `pointermove` event fires whenever the pointer moves over the canvas, even if the user isn't clicking. `isPainting` ensures we only draw when the pointer is actually held down (set to true on `pointerdown`, false on `pointerup`).

*   **Q: Why use Pointer Events (`pointerdown`, etc.) instead of Mouse Events?**
    *   *Answer*: Pointer Events are a unified API handling mouse, touch, and pen inputs.
        *   **Pressure Sensitivity**: They provide `e.pressure`, which is crucial for handwriting apps to vary line thickness or opacity based on how hard the user presses (especially with a stylus).
        *   **Cross-Device**: One event listener works for both desktop mice and mobile touchscreens.

### 3. Data Capture (The Backbone of Recognition)
**Context**: In `frontend/canvas.js`, you are now pushing data to a `strokes` array.

*   **Q: How do you structure the data for potential AI training?**
    *   *Reference*: `point` object in `frontend/canvas.js`.
    *   *Answer*: We store the *trajectory* as a series of normalized 4D points `(x, y, t, p)`.
        *   **Normalized Coordinates (`x`, `y`)**: We store values between 0.0 and 1.0 (e.g., `x / canvas.width`). This makes the data resolution-independent; the AI doesn't care if you drew on a 500px or 1000px wide screen.
        *   **Relative Time (`t`)**: We store the time in milliseconds *relative to the start of the specific stroke* (`now - strokeStartTime`). This captures the speed and rhythm of writing without using large absolute timestamps.
        *   **Pressure (`p`)**: We capture stylus pressure (defaulting to 0.5 if unavailable). This is useful for distinguishing intended strokes from accidental touches.
    *   *Strokes Array*: The data is structured as an array of stroke objects: `[{ stroke_id: 1, points: [...] }, ...]`.

### 4. CSS & Layout
**Context**: `styles.css` uses flexbox and positioning.

*   **Q: How is the toolbar positioned over the canvas?**
    *   *Reference*: `#toolbar` in `styles.css`.
    *   *Answer*: The `.container` is `relative`, and the `#toolbar` is `absolute`. This allows the toolbar to be placed at specific coordinates (`top: 20px`, `left: 20px`) relative to the container, floating "above" the canvas (controlled by `z-index: 10`).

*   **Q: Why `box-sizing: border-box`?**
    *   *Reference*: Line 1 in `styles.css`.
    *   *Answer*: It changes how element width/height is calculated. It includes padding and borders in the element's total width/height, preventing layout breaking when you add padding to an element set to `width: 100%`.

---

## Part 2: Machine Learning & Feature Engineering (Implemented)

These questions relate to `ml/dataset.py` and the data pipeline you have built.

### 1. Data Loading & Feature Engineering
**Context**: The pipeline involves loading raw JSON data and transforming it into tensor-ready features.

*   **Q: How is the data loaded?**
    *   *Reference*: `_load_data` in `ml/dataset.py`.
    *   *Answer*: The loader iterates through all JSON files in the directory and extracts **every** valid stroke (not just the first one). These are flattened into a single training list.

*   **Q: Where does the data come from?**
    *   *Reference*: `ml/synthetic_generator.py`.
    *   *Answer*: Instead of manually collecting thousands of strokes, we use a **Synthetic Data Generator**.
        *   It systematically creates varying shapes (lines, circles, sines, spirals, bezier curves).
        *   It simulates human-like artifacts: vary speed based on curvature (slows down on turns) and pressure (harder on curves).
        *   This generates ~1200 files of diverse "clean" strokes used as ground truth.

*   **Q: What features do you extract from each point?**
    *   *Reference*: `_process_stroke` method in `ml/dataset.py`.
    *   *Answer*: We extract 5 features per point: `[x, y, dt_norm, p_norm, speed]`.
        1.  **Centered Coordinates (`x`, `y`)**: Normalized to be resolution-independent and centered at (0,0).
        2.  **Delta Time (`dt`)**: Time difference between points, normalized.
        3.  **Pressure (`p`)**: Standardized pressure values.
        4.  **Speed**: Calculated from velocity (`sqrt(dx^2 + dy^2)`), helping the model learn stroke dynamics.

*   **Q: How do you handle strokes of different lengths?**
    *   *Reference*: `_resample` method in `ml/dataset.py`.
    *   *Answer*: We use **Linear Interpolation** to resample every stroke to a fixed sequence length (`max_seq_len=128`).
        *   If a stroke has 50 points, we interpolate it to 128.
        *   If it has 200 points, we downsample it to 128.
        *   *Why?* Neural networks (like LSTMs or Transformers) often work best with or require fixed-size input tensors for batch processing.

### 2. Normalization Strategy
**Context**: Handwriting can be large, small, or offset.

*   **Q: How do you normalize the stroke coordinates?**
    *   *Answer*: We use **Aspect Ratio Preserving Normalization**.
        1.  Calculate the bounding box (`min_x`, `max_x`, `min_y`, `max_y`).
        2.  Find the largest dimension: `scale = max(width, height)`.
        3.  Scale both X and Y by this same `scale` factor.
        *   *Why?* If we scaled X and Y independently (part of standard MinMax scaling), a circle would become an oval. We must preserve the shape.

### 3. Data Augmentation (Denoising)
**Context**: The goal is "Smoothing", so you need a way to train the model to remove noise.

*   **Q: How do you generate training data?**
    *   *Reference*: `StrokeDataset.__getitem__` and `_add_synthetic_noise`.
    *   *Answer*: We generate synthetic training pairs on-the-fly.
        *   **Target (Clean)**: The original stroke drawn by the user (assumed to be relatively smooth/intended path).
        *   **Input (Noisy)**: We apply extensive augmentation to the Clean stroke:
            1.  **Random Rotation**: $\pm 15^\circ$ rotation to make the model rotation-invariant.
            2.  **Random Scaling**: $\pm 20\%$ scaling to handle different stroke sizes.
            3.  **Gaussian Noise**: Adding jitter to the coordinate channels.
        *   **Task**: The model learns the mapping $f(Noisy) \rightarrow Clean$.

### 4. Model Architecture
**Context**: You use a Recurrent Neural Network (RNN) to handle the sequential nature of handwriting.

*   **Q: Describe the model architecture.**
    *   *Reference*: `StrokeLSTM` in `ml/model.py`.
    *   *Answer*: We use an **LSTM (Long Short-Term Memory)** network.
        *   **Input**: `(batch_size, seq_len, 5)` — The 5 input features are `[x, y, dt, p, speed]`.
        *   **Hidden Layers**: Two stacked LSTM layers (`num_layers=2`) with `hidden_size=128`. This allows the model to capture complex temporal dependencies.
        *   **Dropout**: We apply `dropout=0.2` between layers to prevent overfitting.
        *   **Output**: A fully connected linear layer maps the final hidden state to 2 coordinates `(x, y)`.

### 5. Training Pipeline
**Context**: You have a complete training script `ml/train.py` that optimizes the model.

*   **Q: What loss function do you use and why?**
    *   *Reference*: `criterion = nn.MSELoss()` in `ml/train.py`.
    *   *Answer*: We use **Mean Squared Error (MSE)**.
        *   Since this is a **regression** problem (predicting continuous x, y coordinates), MSE is the standard choice. It penalizes large deviations from the "Clean" stroke path more severely.

*   **Q: Which optimizer did you choose?**
    *   *Reference*: `torch.optim.Adam` in `ml/train.py`.
    *   *Answer*: **Adam** (Adaptive Moment Estimation).
        *   It typically converges faster than standard SGD because it adapts the learning rate for each parameter.
        *   We use a learning rate of `1e-3`.

*   **Q: What are your hyperparameters?**
    *   *Answer*:
        *   **Batch Size**: 12 (Small enough for stability, large enough for some parallelization).
        *   **Epochs**: 40 (Sufficient for convergence on this dataset size).
        *   **Sequence Length**: 128 (Fixed length for all strokes via interpolation).

*   **Q: Describe one training step.**
    *   *Answer*:
        1.  **Forward Pass**: Pass noisy stroke batch through the LSTM.
        2.  **Calculate Loss**: Compare predicted coordinates vs. clean target coordinates.
        3.  **Zero Gradients**: `optimizer.zero_grad()` to clear old gradients.
        4.  **Backward Pass**: `loss.backward()` computes gradients via backpropagation.
        5.  **Step**: `optimizer.step()` updates model weights.

---

## Part 3: Real-Time Inference (Backend Integration)

These questions relate to `ml/app.py`, `ml/infer.py`, and the connection between frontend and backend.

### 1. Model Deployment & API
**Context**: You are now serving the model via a web server to interact with the frontend.

*   **Q: How did you expose the model to the frontend?**
    *   *Reference*: `ml/app.py`.
    *   *Answer*: I built a lightweight **Flask API**.
        *   It exposes a single endpoint: `POST /smooth-stroke`.
        *   The frontend sends the raw stroke points as JSON.
        *   The backend processes the data, runs the model, and returns the smoothed coordinates.

*   **Q: How do you ensure low-latency inference?**
    *   *Reference*: `ml/export_model.py` and `ml/infer.py`.
    *   *Answer*:
        1.  **TorchScript**: We export the trained PyTorch model to **TorchScript** (`torch.jit.trace`). This serializes the model structure and weights, allowing it to be run in a C++ runtime often faster than standard PyTorch eager mode.
        2.  **Pre-loading**: The model is loaded into memory *once* when the Flask app starts (`apps.py`), not on every request.
        3.  **No Gradients**: We use `torch.no_grad()` to disable gradient calculation, which reduces memory usage and speeds up computation.

*   **Q: Describe the full data flow for a single stroke.**
    *   *Answer*:
        1.  **User draws**: Frontend captures `pointermove` events.
        2.  **Stroke Finish**: On `pointerup`, JavaScript bundles the stroke data.
        3.  **Request**: JS `fetch()` sends a POST request to `http://localhost:5000/smooth-stroke`.
        4.  **Preprocessing**: Flask app receives data -> `DataSet` logic normalizes and resamples features (padding/interpolating to 128 points).
        5.  **Inference**: The LSTM model predicts the smoothed (x, y) coordinates.
        6.  **Response**: The backend sends the smoothed points back to the client.
        7.  **Visualization**: The frontend receives the data, denormalizes it (maps back to screen coordinates), and redraws the smooth stroke on top of the original.

---

## Part 4: Soft Skills / Behavioral

*   **Q: What was the most challenging part of this project?**
    *   *Suggested Answer*: "Ensuring the data pipeline was robust. For example, handling strokes with different point counts and correctly normalizing them without distorting the aspect ratio."
    *   *Alternative Answer*: "Connecting the asynchronous frontend (JavaScript) with the synchronous ML backend (Python). ensuring the coordinate systems matched perfectly (normalization on server vs denormalization on client) was tricky."

*   **Q: How would you improve this app?**
    *   *Idea 1*: **WebAssembly (WASM)**: compiling the PyTorch model (or using ONNX Runtime) to WebAssembly to run inference directly in the browser. This would remove the network latency and the need for a Python backend.
    *   *Idea 2*: **Stroke Classification**: Extending the model to not just *smooth* the stroke, but *recognize* it (e.g., classify it as a letter 'A', 'B', etc.).
    *   *Idea 3*: **Multi-stroke support**: Currently, the ML pipeline processes individual strokes. Extending it to handle full characters (composed of multiple strokes) would be the next logical step.
