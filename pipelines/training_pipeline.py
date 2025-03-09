# pipelines/training_pipeline.py
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
from pipelines.mlflow_elastic import get_es_logger
import time
import json


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
    
    # Get Elasticsearch logger
    es_logger = get_es_logger()
    
    # Start system monitoring if not already started
    es_logger.start_system_monitoring(interval=10)  # More frequent during training

    # Record start time
    start_time = time.time()
    
    run_name = "churn_model_balanced"
    with mlflow.start_run(run_name=run_name) as run:
        print("Training model with class weights.")
        run_id = run.info.run_id

        # Calculate class weights
        class_weights = {0: 1, 1: 5}

        # Initialize model
        model = RandomForestClassifier(
            n_estimators=100, class_weight=class_weights, random_state=42
        )
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Record training time
        training_time = time.time() - start_time

        # Evaluate metrics
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate metrics
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred, pos_label=1)
        test_precision = precision_score(y_test, y_test_pred, pos_label=1)
        test_f1 = f1_score(y_test, y_test_pred, pos_label=1)
        test_roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        
        # Training metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_recall = recall_score(y_train, y_train_pred, pos_label=1)
        train_precision = precision_score(y_train, y_train_pred, pos_label=1)
        train_f1 = f1_score(y_train, y_train_pred, pos_label=1)

        # Create parameters dict
        params = {
            "n_estimators": 100,
            "max_depth": "None",
            "min_samples_split": 2,
            "class_weight": json.dumps(class_weights),  # Serialize as JSON string
            "training_time_seconds": training_time
        }
        
        # Create metrics dict
        metrics = {
            "test_accuracy": test_accuracy,
            "test_recall_churn": test_recall,
            "test_precision_churn": test_precision,
            "test_f1_churn": test_f1,
            "test_roc_auc": test_roc_auc,
            "train_accuracy": train_accuracy,
            "train_recall": train_recall,
            "train_precision": train_precision,
            "train_f1": train_f1,
            "training_time_seconds": training_time
        }

        # Log parameters to MLflow
        mlflow.log_params(params)

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        
        # Get feature importance
        feature_importance = model.feature_importances_
        feature_names = X_train.columns
        
        # Create feature importance dict
        importance_dict = {}
        for i, col in enumerate(feature_names):
            importance_dict[f"importance_{col}"] = float(feature_importance[i])
            # Log top 10 feature importances to MLflow
            if i < 10:
                mlflow.log_metric(f"importance_{col}", float(feature_importance[i]))
        
        # Log to Elasticsearch
        es_logger.log_parameters(run_id, params)
        es_logger.log_metrics(run_id, metrics)
        
        # Log model performance to Elasticsearch
        performance_data = {
            **metrics,
            "model_type": "RandomForestClassifier",
            "feature_count": len(feature_names),
            "top_feature": feature_names[feature_importance.argmax()],
            "top_feature_importance": float(feature_importance.max())
        }
        es_logger.log_model_performance(run_id, "churn_model", performance_data)

        # Save model artifacts
        joblib.dump(model, "churn_model.joblib")
        mlflow.sklearn.log_model(model, "model")
        
        # Log feature importance as artifact
        feature_imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values(by='Importance', ascending=False)
        
        feature_imp_path = "feature_importance.csv"
        feature_imp_df.to_csv(feature_imp_path, index=False)
        mlflow.log_artifact(feature_imp_path)

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

    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Get Elasticsearch logger
    es_logger = get_es_logger()
    
    # Start time
    start_time = time.time()

    run_name = "churn_model_retrain"
    with mlflow.start_run(run_name=run_name) as run:
        print(f"Retraining with: n_estimators={n_estimators}")
        run_id = run.info.run_id

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
        )
        model.fit(X_train, y_train)
        
        # Training time
        training_time = time.time() - start_time
        
        # Parameters dictionary
        params = {
            "n_estimators": n_estimators,
            "max_depth": str(max_depth),
            "min_samples_split": min_samples_split,
            "training_time_seconds": training_time
        }

        # Log parameters to MLflow
        mlflow.log_params(params)
        
        # Calculate metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, pos_label=1)
        precision = precision_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        
        # Metrics dictionary
        metrics = {
            "test_accuracy": accuracy,
            "test_recall": recall,
            "test_precision": precision,
            "test_f1": f1,
            "test_roc_auc": roc_auc,
            "training_time_seconds": training_time
        }
        
        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        
        # Log to Elasticsearch
        es_logger.log_parameters(run_id, params)
        es_logger.log_metrics(run_id, metrics)
        
        # Log model performance to Elasticsearch
        feature_names = X_train.columns
        feature_importance = model.feature_importances_
        
        performance_data = {
            **metrics,
            "model_type": "RandomForestClassifier",
            "feature_count": len(feature_names),
            "top_feature": feature_names[feature_importance.argmax()],
            "top_feature_importance": float(feature_importance.max())
        }
        es_logger.log_model_performance(run_id, "churn_model_retrain", performance_data)

        # Save model
        joblib.dump(model, "churn_model.joblib")
        mlflow.sklearn.log_model(model, "model")

    return model


def main():
    train_model()


if __name__ == "__main__":
    main()
