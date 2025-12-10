"""
Command-line interface for making predictions
"""
import argparse
import logging
import sys
from src.predict import load_model_and_scaler, make_prediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Diabetes Predictor - CLI Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app/cli.py --input "6,148,72,35,0,33.6,0.627,50"
  python app/cli.py --model models/logistic_reg.pkl --input "6,148,72,35,0,33.6,0.627,50"
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input features as comma-separated values'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='models/random_forest.pkl',
        help='Path to the saved model (default: models/random_forest.pkl)'
    )
    parser.add_argument(
        '--scaler', '-s',
        type=str,
        default='models/scaler.pkl',
        help='Path to the saved scaler (default: models/scaler.pkl)'
    )
    
    return parser.parse_args()


def main():
    """Main CLI function."""
    args = parse_args()
    
    try:
        # Parse input
        input_values = [float(x.strip()) for x in args.input.split(',')]
        
        if len(input_values) != len(FEATURE_NAMES):
            print(f"Error: Expected {len(FEATURE_NAMES)} features, "
                  f"got {len(input_values)}")
            print(f"Features: {', '.join(FEATURE_NAMES)}")
            sys.exit(1)
        
        # Load model and scaler
        model, scaler = load_model_and_scaler(args.model, args.scaler)
        
        # Make prediction
        result = make_prediction(input_values, model, scaler, FEATURE_NAMES)
        
        # Display results
        print("\n" + "="*50)
        print("DIABETES PREDICTION RESULTS")
        print("="*50)
        print(f"Input: {', '.join(f'{name}={val}' for name, val in zip(FEATURE_NAMES, input_values))}")
        print("-"*50)
        print(f"Prediction: {'POSITIVE (Diabetes)' if result['prediction'] == 1 else 'NEGATIVE (No Diabetes)'}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        print(f"Diabetes Probability: {result['diabetes_probability']*100:.2f}%")
        print(f"No Diabetes Probability: {result['no_diabetes_probability']*100:.2f}%")
        print("="*50 + "\n")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Invalid input - {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
