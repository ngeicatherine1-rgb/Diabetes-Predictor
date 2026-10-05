"""
Flask web application for diabetes prediction
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.predict import load_model_and_scaler, make_prediction

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "app_config.yaml"
MODEL_DIR = ROOT / "models"


def ensure_model_artifacts():
    """Create a minimal default model if the repository has no serialized artifacts."""
    model_path = MODEL_DIR / "random_forest.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"

    if model_path.exists() and scaler_path.exists():
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    n_samples = 768
    df = pd.DataFrame(
        {
            "Pregnancies": np.random.randint(0, 17, n_samples),
            "Glucose": np.random.randint(44, 200, n_samples),
            "BloodPressure": np.random.randint(24, 122, n_samples),
            "SkinThickness": np.random.randint(0, 99, n_samples),
            "Insulin": np.random.randint(0, 846, n_samples),
            "BMI": np.random.uniform(18.2, 67.1, n_samples),
            "DiabetesPedigreeFunction": np.random.uniform(0.078, 2.42, n_samples),
            "Age": np.random.randint(21, 81, n_samples),
        }
    )
    df["Outcome"] = ((df["Glucose"] > 125) & (df["BMI"] > 30)).astype(int)
    noise_idx = np.random.choice(len(df), 150, replace=False)
    df.loc[noise_idx, "Outcome"] = 1 - df.loc[noise_idx, "Outcome"]
    data_dir = ROOT / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / "diabetes.csv", index=False)

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    scaler.fit(X_train)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(scaler.transform(X_train), y_train)

    import joblib

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)


# Initialize Flask app
app = Flask(__name__, template_folder=str(ROOT / "app" / "templates"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

ensure_model_artifacts()

# Load model and scaler
model, scaler = load_model_and_scaler(
    str(MODEL_DIR / "random_forest.pkl"),
    str(MODEL_DIR / "scaler.pkl"),
)

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


@app.route("/")
def home():
    """Render home page."""
    return render_template("index.html", features=FEATURE_NAMES)


@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint for predictions."""
    try:
        data = request.get_json()

        if not all(feature in data for feature in FEATURE_NAMES):
            return jsonify({"error": "Missing features"}), 400

        input_values = [float(data[feature]) for feature in FEATURE_NAMES]
        result = make_prediction(input_values, model, scaler, FEATURE_NAMES)
        return jsonify(result)

    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    flask_config = config["flask"]
    app.run(
        host=flask_config["host"],
        port=flask_config["port"],
        debug=flask_config["debug"],
        threaded=flask_config["threaded"],
    )
