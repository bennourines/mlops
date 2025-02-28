# model_evaluation.py
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model():
    """Loads the trained model and processed data, then evaluates model performance."""
    print("Loading model and processed data for evaluation...")
    # Load processed data
    data = joblib.load("processed_data.joblib")
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Load the trained model
    model = joblib.load("churn_model.joblib")

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])

    print("✅ Model Evaluation Results:")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(report)

    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n📊 Confusion Matrix:")
    print(f"True Negatives: {cm[0][0]}")
    print(f"False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}")
    print(f"True Positives: {cm[1][1]}")

    return acc, report


def main():
    evaluate_model()  # Appeler la fonction principale


if __name__ == "__main__":
    main()
