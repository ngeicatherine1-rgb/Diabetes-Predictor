"""
Data loading and preprocessing functions
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_data(filepath):
    """
    Load raw CSV data.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        data = pd.read_csv(filepath)
        logger.info(f"Data loaded successfully from {filepath}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def handle_missing_values(data, method='mean'):
    """
    Handle missing values in the dataset.
    
    Args:
        data (pd.DataFrame): Dataset with missing values
        method (str): Method to handle missing values ('mean', 'median', 'drop')
        
    Returns:
        pd.DataFrame: Data with handled missing values
    """
    if method == 'mean':
        data = data.fillna(data.mean())
    elif method == 'median':
        data = data.fillna(data.median())
    elif method == 'drop':
        data = data.dropna()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    logger.info(f"Missing values handled using {method} method")
    return data


def preprocess_data(data, target_column='Outcome', test_size=0.2, random_state=42):
    """
    Preprocess dataset: handle missing values, scale features, split data.
    
    Args:
        data (pd.DataFrame): Raw dataset
        target_column (str): Name of target column
        test_size (float): Proportion of test set
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    # Handle missing values
    data = handle_missing_values(data)
    
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split data. Try stratified split first; if dataset is too small or
    # stratification fails, fall back to a non-stratified split.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError as e:
        logger.warning(
            "Stratified train/test split failed (%s). Falling back to non-stratified split.",
            e,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    logger.info("Data preprocessing completed")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def load_and_preprocess_data(filepath, **kwargs):
    """
    Load and preprocess data in one function.
    
    Args:
        filepath (str): Path to CSV file
        **kwargs: Additional arguments for preprocessing
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    data = load_data(filepath)
    return preprocess_data(data, **kwargs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(
        "data/raw/diabetes.csv"
    )
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
