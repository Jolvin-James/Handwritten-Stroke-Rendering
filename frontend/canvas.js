// frontend/canvas.js
const canvas = document.getElementById("drawing-board");
const toolbar = document.getElementById("toolbar");
const ctx = canvas.getContext("2d");

const canvasOffsetX = canvas.offsetLeft;
const canvasOffsetY = canvas.offsetTop;

canvas.width = window.innerWidth - canvasOffsetX;
canvas.height = window.innerHeight - canvasOffsetY;

let isPainting = false;
let lineWidth = 5;
let exportCount = 1;

// Stroke storage
const strokes = [];
let currentStroke = [];
let strokeStartTime = 0;
let lastPoint = null;

// Toolbar actions
toolbar.addEventListener("click", (e) => {
    if (e.target.id === "clear") {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        strokes.length = 0;
    }

    if (e.target.id === "export") {
        const data = {
            canvas: {
                width: canvas.width,
                height: canvas.height
            },
            strokes: strokes
        };

        const json = JSON.stringify(data, null, 2);
        const blob = new Blob([json], { type: "application/json" });
        const url = URL.createObjectURL(blob);

        const paddedIndex = String(exportCount).padStart(3, "0");
        const filename = `strokes_${paddedIndex}.json`;

        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        a.click();

        URL.revokeObjectURL(url);

        exportCount++;
    }
});

toolbar.addEventListener("change", (e) => {
    if (e.target.id === "stroke") {
        ctx.strokeStyle = e.target.value;
    }

    if (e.target.id === "lineWidth") {
        lineWidth = e.target.value;
    }
});

// Pointer down
canvas.addEventListener("pointerdown", (e) => {
    isPainting = true;
    ctx.beginPath();

    strokeStartTime = performance.now();
    currentStroke = [];
    lastPoint = null;

    const x = e.clientX - canvasOffsetX;
    const y = e.clientY - canvasOffsetY;

    ctx.moveTo(x, y);

    const point = {
        x: x / canvas.width,
        y: y / canvas.height,
        t: 0,
        p: e.pressure || 0.5
    };

    currentStroke.push(point);
    lastPoint = point;
});

// Pointer move
canvas.addEventListener("pointermove", (e) => {
    if (!isPainting) return;

    const x = e.clientX - canvasOffsetX;
    const y = e.clientY - canvasOffsetY;

    ctx.lineWidth = lineWidth;
    ctx.lineCap = "round";
    ctx.lineTo(x, y);
    ctx.stroke();

    const nx = x / canvas.width;
    const ny = y / canvas.height;

    const now = performance.now();
    const t = now - strokeStartTime;

    if (lastPoint) {
        const dx = nx - lastPoint.x;
        const dy = ny - lastPoint.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < 0.001) return;
    }

    const point = {
        x: nx,
        y: ny,
        t: t,
        p: e.pressure || 0.5
    };

    currentStroke.push(point);
    lastPoint = point;
});

// Pointer up
canvas.addEventListener("pointerup", () => {
    isPainting = false;
    ctx.stroke();
    ctx.beginPath();

    if (currentStroke.length > 1) {
        strokes.push({
            stroke_id: strokes.length + 1,
            points: currentStroke
        });
    }

    currentStroke = [];
    lastPoint = null;
});

// Pointer leave safety
canvas.addEventListener("pointerleave", () => {
    isPainting = false;
    ctx.beginPath();
});
