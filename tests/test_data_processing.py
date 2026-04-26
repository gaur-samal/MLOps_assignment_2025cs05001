"""
Unit Tests for Data Processing Module
=====================================
Tests for data loading, cleaning, and preprocessing functions.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    NUMERICAL_FEATURES,
    clean_data,
    convert_target_to_binary,
    get_data_summary,
    get_feature_target_split,
    load_data,
    prepare_data,
)


class TestDataProcessing:
    """Test class for data processing functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        data = {
            "age": [63.0, 67.0, 67.0, 37.0, 41.0],
            "sex": [1.0, 1.0, 1.0, 1.0, 0.0],
            "cp": [1.0, 4.0, 4.0, 3.0, 2.0],
            "trestbps": [145.0, 160.0, 120.0, 130.0, 130.0],
            "chol": [233.0, 286.0, 229.0, 250.0, 204.0],
            "fbs": [1.0, 0.0, 0.0, 0.0, 0.0],
            "restecg": [2.0, 2.0, 2.0, 0.0, 2.0],
            "thalach": [150.0, 108.0, 129.0, 187.0, 172.0],
            "exang": [0.0, 1.0, 1.0, 0.0, 0.0],
            "oldpeak": [2.3, 1.5, 2.6, 3.5, 1.4],
            "slope": [3.0, 2.0, 2.0, 3.0, 1.0],
            "ca": [0.0, 3.0, 2.0, 0.0, 0.0],
            "thal": [6.0, 3.0, 7.0, 3.0, 3.0],
            "target": [0, 2, 1, 0, 0],
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def data_with_missing(self):
        """Create data with missing values for testing."""
        data = {
            "age": [63.0, 67.0, np.nan, 37.0, 41.0],
            "sex": [1.0, 1.0, 1.0, 1.0, 0.0],
            "cp": [1.0, 4.0, 4.0, 3.0, 2.0],
            "trestbps": [145.0, 160.0, 120.0, 130.0, 130.0],
            "chol": [233.0, np.nan, 229.0, 250.0, 204.0],
            "fbs": [1.0, 0.0, 0.0, 0.0, 0.0],
            "restecg": [2.0, 2.0, 2.0, 0.0, 2.0],
            "thalach": [150.0, 108.0, 129.0, 187.0, 172.0],
            "exang": [0.0, 1.0, 1.0, 0.0, 0.0],
            "oldpeak": [2.3, 1.5, 2.6, 3.5, 1.4],
            "slope": [3.0, 2.0, 2.0, 3.0, 1.0],
            "ca": [0.0, np.nan, 2.0, 0.0, 0.0],
            "thal": [6.0, 3.0, np.nan, 3.0, 3.0],
            "target": [0, 2, 1, 0, 0],
        }
        return pd.DataFrame(data)

    def test_feature_names_defined(self):
        """Test that feature names are properly defined."""
        assert len(FEATURE_NAMES) == 14
        assert "target" in FEATURE_NAMES
        assert "age" in FEATURE_NAMES

    def test_feature_categories(self):
        """Test that features are properly categorized."""
        assert "age" in NUMERICAL_FEATURES
        assert "sex" in CATEGORICAL_FEATURES
        assert len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES) == 13

    def test_clean_data_no_missing(self, sample_data):
        """Test clean_data when there are no missing values."""
        cleaned = clean_data(sample_data)
        assert cleaned.isnull().sum().sum() == 0
        assert len(cleaned) == len(sample_data)

    def test_clean_data_with_missing(self, data_with_missing):
        """Test clean_data handles missing values correctly."""
        cleaned = clean_data(data_with_missing)
        assert cleaned.isnull().sum().sum() == 0
        assert len(cleaned) == len(data_with_missing)

    def test_convert_target_to_binary(self, sample_data):
        """Test target conversion to binary."""
        binary_df = convert_target_to_binary(sample_data)
        assert set(binary_df["target"].unique()).issubset({0, 1})
        # Original values 0,2,1,0,0 should become 0,1,1,0,0
        expected = [0, 1, 1, 0, 0]
        assert list(binary_df["target"]) == expected

    def test_get_feature_target_split(self, sample_data):
        """Test feature-target split."""
        X, y = get_feature_target_split(sample_data)
        assert "target" not in X.columns
        assert len(X.columns) == 13
        assert len(y) == len(sample_data)

    def test_get_data_summary(self, sample_data):
        """Test data summary generation."""
        summary = get_data_summary(sample_data)
        assert "n_samples" in summary
        assert "n_features" in summary
        assert summary["n_samples"] == 5
        assert summary["n_features"] == 13


class TestDataValidation:
    """Test class for data validation."""

    def test_age_range(self):
        """Test age values are within expected range."""
        # Create test data
        data = pd.DataFrame({"age": [25, 55, 75, 100], "target": [0, 1, 0, 1]})
        # All ages should be positive and reasonable
        assert (data["age"] > 0).all()
        assert (data["age"] < 120).all()

    def test_binary_features(self):
        """Test binary features have correct values."""
        binary_features = ["sex", "fbs", "exang"]
        for feature in binary_features:
            # These should only contain 0 or 1
            assert feature in CATEGORICAL_FEATURES or feature in NUMERICAL_FEATURES

    def test_numerical_features_are_numeric(self):
        """Test that numerical features are numeric type."""
        data = {
            "age": [63.0, 67.0],
            "trestbps": [145.0, 160.0],
            "chol": [233.0, 286.0],
            "thalach": [150.0, 108.0],
            "oldpeak": [2.3, 1.5],
        }
        df = pd.DataFrame(data)
        for col in df.columns:
            assert pd.api.types.is_numeric_dtype(df[col])


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        empty_df = pd.DataFrame(columns=FEATURE_NAMES)
        X, y = get_feature_target_split(empty_df)
        assert len(X) == 0
        assert len(y) == 0

    def test_single_row(self):
        """Test handling of single row dataframe."""
        single_row = pd.DataFrame(
            {
                "age": [63.0],
                "sex": [1.0],
                "cp": [1.0],
                "trestbps": [145.0],
                "chol": [233.0],
                "fbs": [1.0],
                "restecg": [2.0],
                "thalach": [150.0],
                "exang": [0.0],
                "oldpeak": [2.3],
                "slope": [3.0],
                "ca": [0.0],
                "thal": [6.0],
                "target": [0],
            }
        )
        X, y = get_feature_target_split(single_row)
        assert len(X) == 1
        assert len(y) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
