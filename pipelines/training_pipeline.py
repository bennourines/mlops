# model_training.py
import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def train_model():
    """Loads processed data, trains a RandomForest model, logs with MLflow, and saves the model."""

    # Load processed data
    print("Loading processed data...")
    data = joblib.load("processed_data.joblib")
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")

    run_name = "churn_model_balanced"
    with mlflow.start_run(run_name=run_name):
        print("Training model with class weights.")

        # Calculate class weights
        class_weights = {0: 1, 1: 5}  # Fixed indentation

        # Initialize model
        model = RandomForestClassifier(
            n_estimators=100, class_weight=class_weights, random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate metrics
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate metrics
        test_recall = recall_score(y_test, y_test_pred, pos_label=1)
        test_precision = precision_score(y_test, y_test_pred, pos_label=1)
        test_f1 = f1_score(y_test, y_test_pred, pos_label=1)
        test_roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        # Log parameters (fix class_weight value)
        mlflow.log_params(
            {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "class_weight": "{0: 1, 1: 5}",  # Log actual weights used
            }
        )

        # Log metrics
        mlflow.log_metrics(
            {
                "test_recall_churn": test_recall,
                "test_precision_churn": test_precision,
                "test_f1_churn": test_f1,
                "test_roc_auc": test_roc_auc,
            }
        )

        # Save model
        joblib.dump(model, "churn_model.joblib")
        mlflow.sklearn.log_model(model, "model")

    return model


def retrain_model(
    n_estimators: int = 100,
    max_depth: int = None,
    min_samples_split: int = 2,
):
    """Retrains the model with new hyperparameters."""
    # Load processed data
    data = joblib.load("processed_data.joblib")
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    mlflow.set_tracking_uri("http://localhost:5000")

    run_name = "churn_model_retrain"
    with mlflow.start_run(run_name=run_name):
        print(f"Retraining with: n_estimators={n_estimators}")  # Retirer class_weight

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # Log parameters (sans class_weight)
        mlflow.log_params(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
            }
        )

        # Calculate metrics
        y_pred = model.predict(X_test)
        recall = recall_score(y_test, y_pred, pos_label=1)
        mlflow.log_metric("test_recall", recall)

        # Save model
        joblib.dump(model, "churn_model.joblib")
        mlflow.sklearn.log_model(model, "model")

    return model


def main():
    train_model()


if __name__ == "__main__":
    main()
