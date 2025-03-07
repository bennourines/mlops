# model_evaluation.py
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
from mlflow_elastic import MLflowElasticLogger, get_current_run_id

# Initialize Elasticsearch logger
es_logger = MLflowElasticLogger(es_host="http://localhost:9200", es_index="mlflow-metrics")

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
    report_dict = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"], output_dict=True)
    report_str = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])

    print("✅ Model Evaluation Results:")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(report_str)

    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n📊 Confusion Matrix:")
    print(f"True Negatives: {cm[0][0]}")
    print(f"False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}")
    print(f"True Positives: {cm[1][1]}")

    # Log evaluation metrics to MLflow and Elasticsearch
    mlflow.set_tracking_uri("http://localhost:5000")
    
    with mlflow.start_run(run_name="model_evaluation"):
        run_id = get_current_run_id()
        
        # Extract metrics from classification report and confusion matrix
        metrics = {
            "accuracy": acc,
            "precision_no_churn": report_dict["No Churn"]["precision"],
            "recall_no_churn": report_dict["No Churn"]["recall"],
            "f1_no_churn": report_dict["No Churn"]["f1-score"],
            "precision_churn": report_dict["Churn"]["precision"],
            "recall_churn": report_dict["Churn"]["recall"],
            "f1_churn": report_dict["Churn"]["f1-score"],
            "true_negatives": int(cm[0][0]),
            "false_positives": int(cm[0][1]),
            "false_negatives": int(cm[1][0]),
            "true_positives": int(cm[1][1])
        }
        
        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        
        # Log model evaluation results to Elasticsearch
        es_logger.log_metrics(run_id, metrics)
        es_logger.log_model_performance(run_id, "churn_model_evaluation", metrics)

    return acc, report_str

def main():
    evaluate_model()

if __name__ == "__main__":
    main()
