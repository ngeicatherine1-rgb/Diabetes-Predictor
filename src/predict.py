"""
Prediction functions for new data
"""
import logging
import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)


def load_model_and_scaler(model_path='models/random_forest.pkl', 
                         scaler_path='models/scaler.pkl'):
    """
    Load saved model and scaler.
    
    Args:
        model_path (str): Path to saved model
        scaler_path (str): Path to saved scaler
        
    Returns:
        tuple: (model, scaler)
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    logger.info(f"Model loaded from {model_path}")
    logger.info(f"Scaler loaded from {scaler_path}")
    return model, scaler


def make_prediction(input_data, model, scaler, feature_names=None):
    """
    Make prediction on new data.
    
    Args:
        input_data: Input features (array, list, or dict)
        model: Trained model
        scaler: Fitted scaler
        feature_names (list): Names of features
        
    Returns:
        dict: Prediction results including label and probability
    """
    # Convert input to array if needed
    if isinstance(input_data, dict):
        input_data = np.array([input_data[fname] for fname in feature_names])
    elif isinstance(input_data, list):
        input_data = np.array(input_data)
    
    # Ensure 2D shape
    if input_data.ndim == 1:
        input_data = input_data.reshape(1, -1)
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    result = {
        'prediction': int(prediction),
        'diabetes_probability': float(probability[1]),
        'no_diabetes_probability': float(probability[0]),
        'confidence': float(max(probability)),
    }
    
    logger.info(f"Prediction made: {result['prediction']}, "
                f"Confidence: {result['confidence']:.4f}")
    
    return result


def batch_predict(input_data, model, scaler):
    """
    Make predictions on multiple records.
    
    Args:
        input_data: Input features (array or DataFrame)
        model: Trained model
        scaler: Fitted scaler
        
    Returns:
        np.ndarray: Predictions
    """
    if isinstance(input_data, pd.DataFrame):
        input_data = input_data.values
    
    input_scaled = scaler.transform(input_data)
    predictions = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)
    
    logger.info(f"Batch prediction completed for {len(predictions)} records")
    
    return predictions, probabilities


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage
    model, scaler = load_model_and_scaler()
    
    # Example input (8 features)
    sample_input = [6, 148, 72, 35, 0, 33.6, 0.627, 50]
    result = make_prediction(sample_input, model, scaler)
    print(f"Prediction: {result}")
