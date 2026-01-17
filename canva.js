const canvas = document.getElementById('drawing-board');
const toolbar = document.getElementById('toolbar');
const ctx = canvas.getContext('2d');

const canvasOffsetX = canvas.offsetLeft;
const canvasOffsetY = canvas.offsetTop;

canvas.width = window.innerWidth - canvasOffsetX;
canvas.height = window.innerHeight - canvasOffsetY;

let isPainting = false;
let lineWidth = 5;
let startX;
let startY;

toolbar.addEventListener('click', e => {
    if (e.target.id === 'clear') {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
});

toolbar.addEventListener('change', e => {
    if (e.target.id === 'stroke') {
        ctx.strokeStyle = e.target.value;
    }

    if (e.target.id === 'lineWidth') {
        lineWidth = e.target.value;
    }

});

// Array to store all strokes
const strokes = [];
let currentStroke = [];

const draw = (e) => {
    if (!isPainting) {
        return;
    }

    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';

    // For visual drawing, we keep using the existing logic
    // but we might want to use e.pressure in the future
    ctx.lineTo(e.clientX - canvasOffsetX, e.clientY);
    ctx.stroke();

    // Capture data
    const point = {
        x: e.clientX - canvasOffsetX,
        y: e.clientY - canvasOffsetY,
        time: Date.now(),
        pressure: e.pressure
    };
    currentStroke.push(point);
}

canvas.addEventListener('pointerdown', (e) => {
    isPainting = true;
    startX = e.clientX;
    startY = e.clientY;

    // Start a new stroke
    currentStroke = [];
    // Capture the starting point
    const point = {
        x: e.clientX - canvasOffsetX,
        y: e.clientY - canvasOffsetY,
        time: Date.now(),
        pressure: e.pressure
    };
    currentStroke.push(point);
    // Ensure the visual path starts correctly if needed (optional improvement, but sticking to flow)
});

canvas.addEventListener('pointerup', e => {
    isPainting = false;
    ctx.stroke();
    ctx.beginPath();

    // Save the completed stroke
    if (currentStroke.length > 0) {
        strokes.push(currentStroke);
        console.log('Stroke captured:', currentStroke);
    }
});

canvas.addEventListener('pointermove', draw);
