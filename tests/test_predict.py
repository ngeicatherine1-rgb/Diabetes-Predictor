"""
Unit tests for predict module
"""
import pytest
import numpy as np
import joblib
from unittest.mock import Mock
from src.predict import make_prediction, batch_predict


@pytest.fixture
def mock_model_and_scaler():
    """Create mock model and scaler."""
    model = Mock()
    model.predict = Mock(return_value=np.array([1]))
    model.predict_proba = Mock(return_value=np.array([[0.3, 0.7]]))
    
    scaler = Mock()
    scaler.transform = Mock(return_value=np.array([[1, 2, 3, 4, 5, 6, 7, 8]]))
    
    return model, scaler


def test_make_prediction(mock_model_and_scaler):
    """Test making a single prediction."""
    model, scaler = mock_model_and_scaler
    input_data = [6, 148, 72, 35, 0, 33.6, 0.627, 50]
    
    result = make_prediction(input_data, model, scaler)
    
    assert 'prediction' in result
    assert 'diabetes_probability' in result
    assert 'no_diabetes_probability' in result
    assert 'confidence' in result
    assert result['prediction'] == 1
    assert result['diabetes_probability'] == 0.7


def test_make_prediction_list_input(mock_model_and_scaler):
    """Test prediction with list input."""
    model, scaler = mock_model_and_scaler
    input_data = [6, 148, 72, 35, 0, 33.6, 0.627, 50]
    
    result = make_prediction(input_data, model, scaler)
    assert result['prediction'] in [0, 1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
