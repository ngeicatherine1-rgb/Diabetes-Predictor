"""
Unit tests for data_loader module
"""
import pytest
import pandas as pd
import numpy as np
from src.data_loader import load_data, handle_missing_values, preprocess_data


@pytest.fixture
def sample_data():
    """Create sample diabetes dataset."""
    return pd.DataFrame({
        'Pregnancies': [6, 1, 8, 1, 0],
        'Glucose': [148, 85, 183, 89, 137],
        'BloodPressure': [72, 66, 64, 66, 40],
        'SkinThickness': [35, 29, 0, 23, 35],
        'Insulin': [0, 0, 0, 94, 168],
        'BMI': [33.6, 26.6, 23.3, 28.1, 43.1],
        'DiabetesPedigreeFunction': [0.627, 0.351, 0.672, 0.167, 2.288],
        'Age': [50, 31, 32, 21, 33],
        'Outcome': [1, 0, 1, 0, 1]
    })


def test_handle_missing_values_mean(sample_data):
    """Test handling missing values with mean."""
    sample_data.loc[0, 'Glucose'] = np.nan
    result = handle_missing_values(sample_data, method='mean')
    assert not result.isnull().any().any()


def test_handle_missing_values_drop(sample_data):
    """Test handling missing values by dropping."""
    sample_data.loc[0, 'Glucose'] = np.nan
    result = handle_missing_values(sample_data, method='drop')
    assert len(result) == len(sample_data) - 1


def test_preprocess_data_shape(sample_data):
    """Test that preprocessing returns correct shapes."""
    X_train, X_test, y_train, y_test, scaler = preprocess_data(sample_data)
    
    assert X_train.shape[0] + X_test.shape[0] == len(sample_data)
    assert X_train.shape[1] == 8  # 8 features
    assert len(y_train) + len(y_test) == len(sample_data)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
