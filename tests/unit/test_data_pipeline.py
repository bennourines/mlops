import pytest
from pipelines.data_pipeline import load_and_prepare_data


def test_data_loading():
    """Test if data is loaded correctly."""
    X_train, y_train, X_test, y_test = load_and_prepare_data()

    # Check if data is not empty
    assert len(X_train) > 0, "Training data is empty"
    assert len(y_train) > 0, "Training labels are empty"
    assert len(X_test) > 0, "Test data is empty"
    assert len(y_test) > 0, "Test labels are empty"


def test_feature_columns():
    """Test if feature columns are correct."""
    X_train, _, _, _ = load_and_prepare_data()

    # Use the exact column names from your dataset
    expected_columns = ["Account length", "Total day minutes", "International plan"]
    for col in expected_columns:
        assert col in X_train.columns, f"Missing column: {col}"
