# ml/app.py
from flask import Flask, request, jsonify
from infer import smooth_stroke
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/smooth-stroke", methods=["POST"])
def smooth_stroke_api():
    data = request.get_json()

    if not data or "points" not in data:
        return jsonify({"error": "Invalid input"}), 400

    points = data["points"]

    for p in points:
        if "x" not in p or "y" not in p or "t" not in p:
            return jsonify({"error": "Invalid point format"}), 400

    if len(points) < 2:
        return jsonify({
            "points": [[p["x"], p["y"]] for p in points],
            "meta": None
        })

    try:
        result = smooth_stroke(points)
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Inference error: {e}")
        return jsonify({"error": "Inference failed"}), 500



if __name__ == "__main__":
    # IMPORTANT: disable reloader so model loads once
    app.run(host="0.0.0.0", port=5000, debug=False)
