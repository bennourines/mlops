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
        top_indices = np.argsort(feature_importances)[-5:]
        top_features = [model_features[i] for i in top_indices]
        
        plt.figure(figsize=(12, 6))
        for feature in top_features:
            # Plot distribution for positive class (Churn=1)
            plt.hist(X_test[feature][y_test == 1], alpha=0.5, bins=20, 
                    label=f"{feature} (Churn)")
        
        plt.title('Feature Distribution for Top Features (Churn Customers)')
        plt.xlabel('Feature Value')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        
        # Save feature distribution plot
        feature_dist_path = "feature_distribution.png"
        plt.savefig(feature_dist_path)
        mlflow.log_artifact(feature_dist_path)
        
        # Log prediction errors analysis
        error_indices = np.where(y_pred != y_test)[0]
        error_analysis = {
            "total_errors": len(error_indices),
            "error_rate": len(error_indices) / len(y_test),
            "false_positives": int(fp),
            "false_negatives": int(fn)
        }
        
        # Log error analysis to MLflow
        for key, value in error_analysis.items():
            mlflow.log_metric(key, value)
        
        # Log error analysis to Elasticsearch
        es_logger.log_metrics(run_id, error_analysis)
        
        # Create threshold analysis
        thresholds = np.linspace(0.1, 0.9, 9)
        threshold_metrics = []
        
        for threshold in thresholds:
            y_pred_threshold = (y_pred_proba >= threshold).astype(int)
            precision_t = precision_score(y_test, y_pred_threshold, zero_division=0)
            recall_t = recall_score(y_test, y_pred_threshold, zero_division=0)
            f1_t = f1_score(y_test, y_pred_threshold, zero_division=0)
            
            threshold_metrics.append({
                "threshold": threshold,
                "precision": precision_t,
                "recall": recall_t,
                "f1": f1_t
            })
        
        # Convert to DataFrame for easier plotting
        threshold_df = pd.DataFrame(threshold_metrics)
        
        # Create threshold analysis plot
        plt.figure(figsize=(10, 6))
        plt.plot(threshold_df['threshold'], threshold_df['precision'], 'b-', label='Precision')
        plt.plot(threshold_df['threshold'], threshold_df['recall'], 'g-', label='Recall')
        plt.plot(threshold_df['threshold'], threshold_df['f1'], 'r-', label='F1')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Precision, Recall, and F1 Score vs Threshold')
        plt.legend()
        plt.grid(True)
        
        # Save threshold analysis plot
        threshold_path = "threshold_analysis.png"
        plt.savefig(threshold_path)
        mlflow.log_artifact(threshold_path)
        
        # Log optimal threshold based on F1 score
        optimal_idx = threshold_df['f1'].idxmax()
        optimal_threshold = threshold_df.loc[optimal_idx, 'threshold']
        mlflow.log_metric("optimal_threshold", optimal_threshold)
        
        # Write results summary to text file
        with open("evaluation_summary.txt", "w") as f:
            f.write(f"MODEL EVALUATION SUMMARY\n")
            f.write(f"=======================\n\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"ROC AUC: {roc_auc:.4f}\n")
            f.write(f"PR AUC: {pr_auc:.4f}\n\n")
            f.write(f"Churn Class Metrics:\n")
            f.write(f"  Precision: {precision_churn:.4f}\n")
            f.write(f"  Recall: {recall_churn:.4f}\n")
            f.write(f"  F1 Score: {f1_churn:.4f}\n\n")
            f.write(f"Confusion Matrix:\n")
            f.write(f"  True Negatives: {tn}\n")
            f.write(f"  False Positives: {fp}\n")
            f.write(f"  False Negatives: {fn}\n")
            f.write(f"  True Positives: {tp}\n\n")
            f.write(f"Optimal Threshold: {optimal_threshold:.2f}\n")
            f.write(f"Evaluation Time: {eval_time:.2f} seconds\n")
        
        # Log summary as artifact
        mlflow.log_artifact("evaluation_summary.txt")
        
        print(f"✅ Model evaluation complete. Results logged to MLflow and Elasticsearch.")
        print(f"   Accuracy: {acc:.4f}, ROC AUC: {roc_auc:.4f}, F1 (Churn): {f1_churn:.4f}")
        
        return metrics


def main():
    evaluate_model()


if __name__ == "__main__":
    main()
