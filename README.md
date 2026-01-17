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
2.  Open `index.html` in any modern web browser.
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
