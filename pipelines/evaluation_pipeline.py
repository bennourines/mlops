# pipelines/evaluation_pipeline.py
import joblib
import mlflow
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc
)
import matplotlib.pyplot as plt
import pandas as pd
from pipelines.mlflow_elastic import get_es_logger
import json
import time


def evaluate_model():
    """Loads the trained model and processed data, then evaluates model performance."""
    print("Loading model and processed data for evaluation...")
    
    # Start time
    start_time = time.time()
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Get Elasticsearch logger
    es_logger = get_es_logger()
    
    # Load processed data
    data = joblib.load("processed_data.joblib")
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Load the trained model
    model = joblib.load("churn_model.joblib")

    with mlflow.start_run(run_name="model_evaluation") as run:
        run_id = run.info.run_id
        
        # Make predictions on the test set
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"], output_dict=True)
        
        # Extract metrics from the report
        precision_churn = report["Churn"]["precision"]
        recall_churn = report["Churn"]["recall"]
        f1_churn = report["Churn"]["f1-score"]
        precision_no_churn = report["No Churn"]["precision"]
        recall_no_churn = report["No Churn"]["recall"]
        f1_no_churn = report["No Churn"]["f1-score"]
        
        # Calculate ROC and PR curve data
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # Evaluation time
        eval_time = time.time() - start_time
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Log metrics to MLflow
        metrics = {
            "accuracy": acc,
            "precision_churn": precision_churn,
            "recall_churn": recall_churn,
            "f1_churn": f1_churn,
            "precision_no_churn": precision_no_churn,
            "recall_no_churn": recall_no_churn,
            "f1_no_churn": f1_no_churn,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "evaluation_time_seconds": eval_time
        }
        
        # Log metrics to MLflow
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        
        # Log to Elasticsearch
        es_logger.log_metrics(run_id, metrics)
        
        # Create ROC curve plot
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        # Save ROC curve plot
        roc_path = "roc_curve.png"
        plt.savefig(roc_path)
        mlflow.log_artifact(roc_path)
        
        # Create PR curve plot
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        # Save PR curve plot
        pr_path = "pr_curve.png"
        plt.savefig(pr_path)
        mlflow.log_artifact(pr_path)
        
        # Create confusion matrix plot
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["No Churn", "Churn"], rotation=45)
        plt.yticks(tick_marks, ["No Churn", "Churn"])
        
        # Add text annotations to the confusion matrix
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Save confusion matrix plot
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        
        # Create feature distribution plot for top 5 features
        model_features = X_test.columns
        feature_importances = model.feature_importances_
        top_indices =
