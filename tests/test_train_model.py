"""
Unit tests for train_model module
"""
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from src.train_model import train_random_forest, train_logistic_regression


@pytest.fixture
def sample_data():
    """Create sample training data."""
    X = np.random.rand(100, 8)
    y = np.random.randint(0, 2, 100)
    return X, y


def test_train_random_forest(sample_data):
    """Test random forest training."""
    X, y = sample_data
    model = train_random_forest(X, y, n_estimators=10, random_state=42)
    
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 10
    
    # Test prediction
    pred = model.predict(X[:5])
    assert len(pred) == 5
    assert all(p in [0, 1] for p in pred)


def test_train_logistic_regression(sample_data):
    """Test logistic regression training."""
    X, y = sample_data
    model = train_logistic_regression(X, y, random_state=42)
    
    assert isinstance(model, LogisticRegression)
    
    # Test prediction
    pred = model.predict(X[:5])
    assert len(pred) == 5
    assert all(p in [0, 1] for p in pred)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
