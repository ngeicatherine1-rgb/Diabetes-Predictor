"""
Model training pipeline
"""
import logging
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path='configs/model_config.yaml'):
    """Load model configuration from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_logistic_regression(X_train, y_train, **params):
    """
    Train Logistic Regression model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        **params: Model parameters
        
    Returns:
        model: Trained model
    """
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    logger.info("Logistic Regression model trained")
    return model


def train_random_forest(X_train, y_train, **params):
    """
    Train Random Forest model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        **params: Model parameters
        
    Returns:
        model: Trained model
    """
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    logger.info("Random Forest model trained")
    return model


def save_model(model, filepath):
    """
    Save trained model to file.
    
    Args:
        model: Trained model
        filepath (str): Path to save model
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load trained model from file.
    
    Args:
        filepath (str): Path to model file
        
    Returns:
        model: Loaded model
    """
    model = joblib.load(filepath)
    logger.info(f"Model loaded from {filepath}")
    return model


def train_and_save_model(X_train, y_train, model_type='random_forest', 
                        output_path='models/', config_path='configs/model_config.yaml'):
    """
    Train and save a model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_type (str): Type of model ('logistic_regression' or 'random_forest')
        output_path (str): Directory to save model
        config_path (str): Path to config file
        
    Returns:
        model: Trained model
    """
    config = load_config(config_path)
    
    if model_type == 'logistic_regression':
        params = config['logistic_regression']
        model = train_logistic_regression(X_train, y_train, **params)
        filename = 'logistic_reg.pkl'
    elif model_type == 'random_forest':
        params = config['random_forest']
        model = train_random_forest(X_train, y_train, **params)
        filename = 'random_forest.pkl'
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    filepath = f"{output_path}{filename}"
    save_model(model, filepath)
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data_loader import load_and_preprocess_data
    
    # Example usage
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data("data/raw/diabetes.csv")
    
    # Train models
    model = train_and_save_model(X_train, y_train, model_type='random_forest')
    print("Model training complete!")
