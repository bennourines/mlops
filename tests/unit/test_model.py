from pipelines.data_pipeline import load_and_prepare_data
from pipelines.training_pipeline import train_model
from sklearn.ensemble import RandomForestClassifier
import urllib3

def test_model_training():
    """Test if the model is trained successfully."""
    model = train_model()
    assert isinstance(
        model, RandomForestClassifier
    ), "Model is not a RandomForestClassifier"


def test_model_accuracy():
    """Test if the model achieves reasonable accuracy."""
    model = train_model()
    _, _, X_test, y_test = load_and_prepare_data()  # Now works

    # Predict and check accuracy
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    assert accuracy > 0.8, "Model accuracy is too low"
