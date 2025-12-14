from src.predict import load_model_and_scaler, make_prediction

FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]


def main():
    model_path = "models/random_forest.pkl"
    scaler_path = "models/scaler.pkl"

    try:
        model, scaler = load_model_and_scaler(model_path, scaler_path)
    except FileNotFoundError as e:
        print(f"Model or scaler file not found: {e}")
        print("Please train models or place the files under the `models/` directory.")
        return

    # Example input values
    sample_input = [6, 148, 72, 35, 0, 33.6, 0.627, 50]

    result = make_prediction(
        sample_input, model, scaler, feature_names=FEATURE_NAMES
    )

    print("\nPrediction result:\n")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
