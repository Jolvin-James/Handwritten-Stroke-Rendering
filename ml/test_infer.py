# ml/test_infer.py
import json
from infer import smooth_stroke

with open("../data/raw/strokes_001.json", "r") as f:
    data = json.load(f)

stroke = data["strokes"][0]["points"]

smoothed = smooth_stroke(stroke)

print("Input points:", len(stroke))
print("Output points:", len(smoothed))
