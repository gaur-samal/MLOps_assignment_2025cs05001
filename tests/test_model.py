"""
Unit Tests for Model Module
===========================
Tests for model training, evaluation, and prediction functions.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import (
    get_model, evaluate_model, cross_validate_model,
    MODEL_CONFIGS
)
from src.predict import (
    HeartDiseasePredictor, create_sample_input,
    FEATURE_NAMES, FEATURE_DESCRIPTIONS
)


class TestModelFunctions:
    """Test class for model functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 13)
        y = np.random.randint(0, 2, n_samples)
        return X, y
    
    @pytest.fixture
    def trained_model(self, sample_data):
        """Create and train a simple model."""
        X, y = sample_data
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        return model
    
    def test_get_model_logistic_regression(self):
        """Test get_model returns correct model type."""
        model = get_model('logistic_regression')
        assert isinstance(model, LogisticRegression)
    
    def test_get_model_random_forest(self):
        """Test get_model returns RandomForest."""
        model = get_model('random_forest')
        assert isinstance(model, RandomForestClassifier)
    
    def test_get_model_invalid(self):
        """Test get_model raises error for invalid model name."""
        with pytest.raises(ValueError):
            get_model('invalid_model')
    
    def test_model_configs_exist(self):
        """Test that model configs are properly defined."""
        assert 'logistic_regression' in MODEL_CONFIGS
        assert 'random_forest' in MODEL_CONFIGS
        assert 'gradient_boosting' in MODEL_CONFIGS
    
    def test_evaluate_model(self, trained_model, sample_data):
        """Test model evaluation returns correct metrics."""
        X, y = sample_data
        metrics = evaluate_model(trained_model, X, y)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
        
        # Check metric ranges
        for metric, value in metrics.items():
            assert 0 <= value <= 1, f"{metric} should be between 0 and 1"
    
    def test_cross_validate_model(self, sample_data):
        """Test cross-validation returns correct structure."""
        X, y = sample_data
        model = LogisticRegression(max_iter=1000, random_state=42)
        
        results = cross_validate_model(model, X, y, cv=3)
        
        assert 'accuracy' in results
        assert 'mean' in results['accuracy']
        assert 'std' in results['accuracy']
        assert 'scores' in results['accuracy']


class TestPredictor:
    """Test class for HeartDiseasePredictor."""
    
    def test_create_sample_input(self):
        """Test sample input creation."""
        sample = create_sample_input()
        
        assert isinstance(sample, dict)
        for feature in FEATURE_NAMES:
            assert feature in sample
    
    def test_feature_names(self):
        """Test feature names are correctly defined."""
        assert len(FEATURE_NAMES) == 13
        assert 'age' in FEATURE_NAMES
        assert 'target' not in FEATURE_NAMES
    
    def test_feature_descriptions(self):
        """Test feature descriptions are complete."""
        for feature in FEATURE_NAMES:
            assert feature in FEATURE_DESCRIPTIONS
    
    def test_predictor_validation(self):
        """Test input validation."""
        predictor = HeartDiseasePredictor()
        
        # Valid input
        valid_input = create_sample_input()
        errors = predictor.validate_input(valid_input)
        assert len(errors) == 0
        
        # Missing feature
        invalid_input = {'age': 50}
        errors = predictor.validate_input(invalid_input)
        assert len(errors) > 0
    
    def test_predictor_validation_invalid_age(self):
        """Test validation catches invalid age."""
        predictor = HeartDiseasePredictor()
        
        invalid_input = create_sample_input()
        invalid_input['age'] = 150  # Invalid age
        
        errors = predictor.validate_input(invalid_input)
        assert any('age' in e.lower() for e in errors)
    
    def test_predictor_validation_invalid_sex(self):
        """Test validation catches invalid sex value."""
        predictor = HeartDiseasePredictor()
        
        invalid_input = create_sample_input()
        invalid_input['sex'] = 2  # Invalid sex
        
        errors = predictor.validate_input(invalid_input)
        assert any('sex' in e.lower() for e in errors)


class TestModelPredictions:
    """Test model prediction quality."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple trained model for testing."""
        np.random.seed(42)
        X = np.random.randn(200, 13)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple decision boundary
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        return model
    
    def test_model_returns_binary_prediction(self, simple_model):
        """Test model returns binary predictions."""
        X_test = np.random.randn(10, 13)
        predictions = simple_model.predict(X_test)
        
        assert all(p in [0, 1] for p in predictions)
    
    def test_model_returns_probabilities(self, simple_model):
        """Test model returns valid probabilities."""
        X_test = np.random.randn(10, 13)
        probas = simple_model.predict_proba(X_test)
        
        # Check shape
        assert probas.shape == (10, 2)
        
        # Check probabilities sum to 1
        for proba in probas:
            assert abs(sum(proba) - 1.0) < 1e-6
        
        # Check probabilities are valid
        assert (probas >= 0).all()
        assert (probas <= 1).all()
    
    def test_model_reproducibility(self, simple_model):
        """Test model predictions are reproducible."""
        X_test = np.random.randn(5, 13)
        
        pred1 = simple_model.predict(X_test)
        pred2 = simple_model.predict(X_test)
        
        assert (pred1 == pred2).all()


class TestInputValidation:
    """Test input validation edge cases."""
    
    def test_boundary_values(self):
        """Test boundary values for features."""
        predictor = HeartDiseasePredictor()
        
        # Test boundary values
        boundary_input = {
            'age': 1,  # Min age
            'sex': 0,
            'cp': 1,
            'trestbps': 50,  # Min BP
            'chol': 100,  # Min chol
            'fbs': 0,
            'restecg': 0,
            'thalach': 60,  # Min HR
            'exang': 0,
            'oldpeak': 0,  # Min
            'slope': 1,
            'ca': 0,
            'thal': 3
        }
        
        errors = predictor.validate_input(boundary_input)
        assert len(errors) == 0
    
    def test_max_boundary_values(self):
        """Test maximum boundary values."""
        predictor = HeartDiseasePredictor()
        
        max_input = {
            'age': 120,  # Max age
            'sex': 1,
            'cp': 4,
            'trestbps': 250,  # Max BP
            'chol': 600,  # Max chol
            'fbs': 1,
            'restecg': 2,
            'thalach': 220,  # Max HR
            'exang': 1,
            'oldpeak': 10,  # Max
            'slope': 3,
            'ca': 3,
            'thal': 7
        }
        
        errors = predictor.validate_input(max_input)
        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
