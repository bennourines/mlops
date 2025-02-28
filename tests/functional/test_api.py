import pytest
import joblib
import pandas as pd
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app import app  # Ensure the import is correct


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    model.predict.return_value = [0]  # 0 for "Will Not Churn"
    return model


@pytest.fixture
def mock_data():
    """Create mock feature data for testing."""
    # This should match the expected features in your model
    return {
        "X_train": pd.DataFrame(
            columns=[
                "account_length",
                "area_code",
                "international_plan",
                "voice_mail_plan",
                "number_vmail_messages",
                "total_day_minutes",
                "total_day_calls",
                "total_eve_minutes",
                "total_eve_calls",
                "total_night_minutes",
                "total_night_calls",
                "total_intl_minutes",
                "total_intl_calls",
                "customer_service_calls",
                "State_OH",  # Add a state column
            ]
        )
    }


@pytest.fixture
def client(mock_model, mock_data):
    """Create a test client with mocked dependencies."""
    # Patch the global variables
    with patch("app.model", mock_model), patch(
        "app.feature_names", mock_data["X_train"].columns.tolist()
    ), patch("app.all_states", ["State_OH"]):
        yield TestClient(app)


def test_predict_endpoint(client):
    """Test the /predict endpoint."""
    response = client.post(
        "/predict",
        json={
            "state": "OH",
            "account_length": 128,
            "area_code": 415,
            "international_plan": "no",
            "voice_mail_plan": "yes",
            "number_vmail_messages": 10,
            "total_day_minutes": 200,
            "total_day_calls": 50,
            "total_eve_minutes": 150,
            "total_eve_calls": 40,
            "total_night_minutes": 100,
            "total_night_calls": 30,
            "total_intl_minutes": 10,
            "total_intl_calls": 5,
            "customer_service_calls": 2,
        },
    )

    print("📌 Réponse API:", response.json())  # Debug
    assert response.status_code == 200, f"Erreur API: {response.json()}"
    assert "prediction" in response.json()
