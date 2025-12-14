"""
Flask web application for diabetes prediction
"""

import logging
import yaml
from flask import Flask, render_template, request, jsonify
from src.predict import load_model_and_scaler, make_prediction

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
with open("configs/app_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load model and scaler
model, scaler = load_model_and_scaler(
    config["models"]["model_path"] + "random_forest.pkl",
    config["models"]["scaler_path"],
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

        # Validate input
        if not all(feature in data for feature in FEATURE_NAMES):
            return jsonify({"error": "Missing features"}), 400

        # Extract values
        input_values = [float(data[feature]) for feature in FEATURE_NAMES]

        # Make prediction
        result = make_prediction(input_values, model, scaler, FEATURE_NAMES)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


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
