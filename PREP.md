# Interview Preparation Guide: Handwritten Stroke Recognition

This guide covers potential interview questions and concepts based on your current codebase. Since your project is currently a **Frontend Drawing Application**, questions will focus heavily on **HTML5 Canvas**, **JavaScript Events**, and **Web Development Basics**.

We also include a section on **Machine Learning & Backend Integration**, which is the logical next step for a "Recognition" app.

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

### 1. Feature Engineering
**Context**: You don't just feed raw pixels to the model; you process the stroke points first.

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
        *   **Input (Noisy)**: We add Gaussian noise to the coordinate channels of the Clean stroke.
        *   **Task**: The model learns the mapping $f(Noisy) \rightarrow Clean$.

---

## Part 3: Soft Skills / Behavioral

*   **Q: What was the most challenging part of this project?**
    *   *Suggested Answer*: "Ensuring the data pipeline was robust. For example, handling strokes with different point counts and correctly normalizing them without distorting the aspect ratio."
*   **Q: How would you improve this app?**
    *   *Idea 1*: **Train the Model**: Now that the dataset pipeline is ready, I would design a Transformer or LSTM-based Autoencoder to actually interpret the data.
    *   *Idea 2*: **Real-time Inference**: Connect the Python backend to the frontend via WebSockets or a REST API to smooth strokes in real-time as the user draws.
    *   *Idea 3*: **Multi-stroke support**: Currently, the ML pipeline processes individual strokes. Extending it to handle full characters (multiple strokes) would be the next logic step.
