import os
from pathlib import Path

import yaml
from flask import Flask, jsonify, request

from src.predict import make_prediction

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "app_config.yaml"
MODEL_PATH = ROOT / "models" / "random_forest.pkl"
SCALER_PATH = ROOT / "models" / "scaler.pkl"

app = Flask(__name__)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}

# Importing the model lazily keeps the startup path cleaner and makes Vercel
# deploys more reliable, especially when the app is cold-started.
model = None
scaler = None
FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


def load_runtime_artifacts():
    global model, scaler
    if model is not None and scaler is not None:
        return

    import joblib

    model = joblib.load(str(MODEL_PATH))
    scaler = joblib.load(str(SCALER_PATH))


@app.get("/")
def home():
    return jsonify({
        "status": "ok",
        "service": "diabetes-predictor",
        "message": "Flask app is running on Vercel."
    })


@app.get("/health")
def health():
    return jsonify({"status": "healthy"})


@app.post("/predict")
def predict():
    try:
        load_runtime_artifacts()
        payload = request.get_json(silent=True) or {}

        missing = [feature for feature in FEATURE_NAMES if feature not in payload]
        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400

        values = [float(payload[feature]) for feature in FEATURE_NAMES]
        result = make_prediction(values, model, scaler, FEATURE_NAMES)
        return jsonify(result)
    except Exception as exc:  # pragma: no cover - kept for deployment clarity
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    host = config.get("flask", {}).get("host", "0.0.0.0")
    port = int(config.get("flask", {}).get("port", 5000))
    app.run(host=host, port=port, debug=False)
