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

const renderedStrokes = [];

// ML RENDER HELPERS 
function denormalizePoints(mlPoints, meta) {
    const { min_x, min_y, scale, center_x, center_y } = meta;

    return mlPoints.map(p => ({
        x: (p[0] + center_x) * scale + min_x,
        y: (p[1] + center_y) * scale + min_y
    }));
}

function drawStroke(points) {
    if (points.length < 2) return;

    ctx.beginPath();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = lineWidth * 0.9;

    ctx.moveTo(
        points[0].x * canvas.width,
        points[0].y * canvas.height
    );

    for (let i = 1; i < points.length; i++) {
        ctx.lineTo(
            points[i].x * canvas.width,
            points[i].y * canvas.height
        );
    }

    ctx.stroke();
}

async function runMLInference(strokePoints) {
    const response = await fetch("http://localhost:5000/smooth-stroke", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ points: strokePoints })
    });

    return await response.json();
}


// Toolbar actions
toolbar.addEventListener("click", (e) => {
    if (e.target.id === "clear") {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        strokes.length = 0;
        renderedStrokes.length = 0;
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
        p: e.pressure > 0 ? e.pressure : 1.0
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
        p: e.pressure > 0 ? e.pressure : 1.0
    };

    currentStroke.push(point);
    lastPoint = point;
});

// Pointer up
canvas.addEventListener("pointerup", async () => {
    isPainting = false;
    ctx.stroke();
    ctx.beginPath();

    if (currentStroke.length < 2) {
        currentStroke = [];
        lastPoint = null;
        return;
    }

    const rawStroke = {
        stroke_id: strokes.length + 1,
        points: currentStroke
    };
    strokes.push(rawStroke);

    try {
        const result = await runMLInference(currentStroke);
        if (!result.points || !Array.isArray(result.points)) {
            throw new Error("Invalid ML response");
        }
        const smoothPoints = denormalizePoints(result.points, result.meta);

        renderedStrokes.push(smoothPoints);
        rawStroke.smoothed = smoothPoints;

        // Redraw everything cleanly
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        renderedStrokes.forEach(stroke => {
            drawStroke(stroke);
        });

        // Optional overlay for demo
        /*
        ctx.setLineDash([4, 4]);
        drawStroke(currentStroke);
        ctx.setLineDash([]);
        */

    } catch (err) {
        console.error("ML inference failed:", err);
        drawStroke(currentStroke);
    }

    currentStroke = [];
    lastPoint = null;
});

// Pointer leave safety
canvas.addEventListener("pointerleave", () => {
    isPainting = false;
    ctx.beginPath();
});

canvas.addEventListener("pointercancel", () => {
    isPainting = false;
    ctx.beginPath();
    currentStroke = [];
    lastPoint = null;
});

window.addEventListener("resize", () => {
    canvas.width = window.innerWidth - canvasOffsetX;
    canvas.height = window.innerHeight - canvasOffsetY;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    renderedStrokes.forEach(stroke => drawStroke(stroke));
});
