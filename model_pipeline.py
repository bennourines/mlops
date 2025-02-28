import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import os
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import sqlite3

warnings.filterwarnings("ignore")


def prepare_data(train_path="churn_80.csv", test_path="churn_20.csv"):
    """Loads, cleans, and prepares data for training and evaluation."""
    print(f"Loading data from {train_path} and {test_path}")

    # Check if files exist
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Data files not found. Please ensure {train_path} and {test_path} exist."
        )

    df_80 = pd.read_csv(train_path)
    df_20 = pd.read_csv(test_path)

    # Fill missing values with mean for numeric columns
    for col in df_80.select_dtypes(include=["float64", "int64"]).columns:
        df_80[col].fillna(df_80[col].mean(), inplace=True)
        df_20[col].fillna(df_20[col].mean(), inplace=True)

    # Encode categorical features
    categorical_features = ["International plan", "Voice mail plan"]
    encoder = OrdinalEncoder()
    df_80[categorical_features] = encoder.fit_transform(df_80[categorical_features])
    df_20[categorical_features] = encoder.transform(df_20[categorical_features])

    # One-hot encode "State"
    df_80 = pd.get_dummies(df_80, columns=["State"], prefix="State")
    df_20 = pd.get_dummies(df_20, columns=["State"], prefix="State")

    # Align test set columns with training set
    df_20 = df_20.reindex(columns=df_80.columns, fill_value=0)

    # Normalize data
    scaler = MinMaxScaler()
    df_80_scaled = pd.DataFrame(scaler.fit_transform(df_80), columns=df_80.columns)
    df_20_scaled = pd.DataFrame(scaler.transform(df_20), columns=df_20.columns)

    # Drop redundant features
    redundant_features = [
        "Total day charge",
        "Total eve charge",
        "Total night charge",
        "Total intl charge",
    ]
    df_80_scaled.drop(columns=redundant_features, inplace=True, errors="ignore")
    df_20_scaled.drop(columns=redundant_features, inplace=True, errors="ignore")

    # Save preprocessors for inference pipeline
    joblib.dump(encoder, "encoder.joblib")
    joblib.dump(scaler, "scaler.joblib")

    # Separate features and labels
    X_train = df_80_scaled.drop(columns=["Churn"])
    y_train = df_80_scaled["Churn"]
    X_test = df_20_scaled.drop(columns=["Churn"])
    y_test = df_20_scaled["Churn"]

    print(
        f"Data prepared: X_train shape: {X_train.shape}, X_test shape: {X_test.shape}"
    )
    return X_train, y_train, X_test, y_test


def train_model(
    X_train,
    y_train,
    X_test,
    y_test,
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
):
    """Trains a RandomForest model and logs with MLflow."""

    # Set tracking URI to the SQLite backend and artifact location
    try:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        print("Connected to MLflow SQLite tracking database")
    except Exception as e:
        print(f"Warning: Could not connect to MLflow SQLite database: {e}")
        print("Defaulting to local tracking")

    # Generate unique model name
    model_name = f"churn_model_{n_estimators}_{max_depth}_{min_samples_split}"

    # Start MLflow run with explicit artifact location
    with mlflow.start_run(run_name=model_name):  # <-- ADD THIS
        print(
            f"Training model with parameters: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}"
        )

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
        )

        # Train model
        model.fit(X_train, y_train)

        # Log hyperparameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)

        # Feature importance
        feature_importance = pd.DataFrame(
            {"Feature": X_train.columns, "Importance": model.feature_importances_}
        ).sort_values("Importance", ascending=False)

        # Log top 10 feature importances
        for feature, importance in feature_importance.head(10).values:
            mlflow.log_param(f"importance_{feature}", importance)

        # Log metrics
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)

        # Get classification report metrics
        report = classification_report(y_test, y_test_pred, output_dict=True)

        # More robust approach to handle the classification report
        for class_label in ["1", 1]:  # Try both string and integer keys
            if str(class_label) in report:
                mlflow.log_metric(
                    "precision_class1", report[str(class_label)]["precision"]
                )
                mlflow.log_metric("recall_class1", report[str(class_label)]["recall"])
                mlflow.log_metric(
                    "f1_score_class1", report[str(class_label)]["f1-score"]
                )
                break
        else:  # This else belongs to the for loop (executes if no break)
            print("Warning: No positive class predictions found in test set")
            mlflow.log_metric("precision_class1", 0.0)
            mlflow.log_metric("recall_class1", 0.0)
            mlflow.log_metric("f1_score_class1", 0.0)

        # Log the model
        mlflow.sklearn.log_model(model, "model")

        # Save locally
        model_filename = f"{model_name}.joblib"
        joblib.dump(model, model_filename)

        # Also save as the default model if test accuracy is good (optional)
        if test_acc > 0.9:
            joblib.dump(model, "churn_model.joblib")
            print(
                f"✅ Model saved as default model due to high accuracy: {test_acc:.4f}"
            )

        print(f"✅ Model trained and saved as {model_filename}")
        print(f"   Train accuracy: {train_acc:.4f}")
        print(f"   Test accuracy: {test_acc:.4f}")

        # Log run_id for reference
        run_id = mlflow.active_run().info.run_id
        print(f"   MLflow run ID: {run_id}")

    return model


def retrain_model(n_estimators=100, max_depth=None, min_samples_split=2):
    """Retrains the model with new hyperparameters."""
    print("Retraining model with new hyperparameters...")
    X_train, y_train, X_test, y_test = prepare_data()

    return train_model(
        X_train,
        y_train,
        X_test,
        y_test,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
    )


def evaluate_model(model, X_test, y_test):
    """Evaluates the model on test data and prints metrics."""
    if model is None:
        raise ValueError("Model is None. Please provide a valid model for evaluation.")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"✅ Model Evaluation Results:")
    print(f"   Accuracy: {acc:.4f}")
    print("\n🔍 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # Get confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n📊 Confusion Matrix:")
    print(f"   True Negatives: {cm[0][0]}")
    print(f"   False Positives: {cm[0][1]}")
    print(f"   False Negatives: {cm[1][0]}")
    print(f"   True Positives: {cm[1][1]}")

    return acc


def save_model(model, filename="churn_model.joblib"):
    """Saves the given model to a file."""
    try:
        joblib.dump(model, filename)
        print(f"💾 Model saved as {filename}")
        return True
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return False


def load_model(filename="churn_model.joblib"):
    """Loads the trained model."""
    try:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file {filename} not found")
        model = joblib.load(filename)
        print(f"📂 Model loaded from {filename}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def predict_single(data, model=None):
    """Make a prediction for a single customer."""
    if model is None:
        model = load_model()

    # Assume data is a dictionary with customer features
    # In a real implementation, you would need to preprocess this data
    # (encoding, scaling, etc.) consistent with the training pipeline
    return model.predict(pd.DataFrame([data]))[0]


def query_mlflow_db():
    """Query the MLflow SQLite database to show experiment information."""
    try:
        conn = sqlite3.connect("mlflow.db")
        cursor = conn.cursor()

        print("\n📊 MLflow Database Summary:")

        # Get experiments
        cursor.execute("SELECT experiment_id, name FROM experiments")
        experiments = cursor.fetchall()
        print(f"\nFound {len(experiments)} experiments:")
        for exp_id, exp_name in experiments:
            print(f"  • Experiment {exp_id}: {exp_name}")

            # Get runs for this experiment
            cursor.execute(
                "SELECT run_uuid, status, start_time, end_time, artifact_uri FROM runs WHERE experiment_id = ?",
                (exp_id,),
            )
            runs = cursor.fetchall()
            print(f"    - Contains {len(runs)} runs")

            # Get metrics for the latest 3 runs
            for i, (run_id, status, start_time, end_time, artifact_uri) in enumerate(
                runs[:3]
            ):
                cursor.execute(
                    "SELECT key, value FROM metrics WHERE run_uuid = ? AND key = 'test_accuracy'",
                    (run_id,),
                )
                metrics = cursor.fetchall()
                metrics_str = ", ".join([f"{k}: {v}" for k, v in metrics])
                print(
                    f"      Run {i+1}: {run_id[:8]}... | Status: {status} | Metrics: {metrics_str}"
                )

        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error querying MLflow database: {e}")
        if "no such table" in str(e).lower():
            print("   This likely means the MLflow database hasn't been created yet.")
            print(
                "   Start the MLflow server with the SQLite backend and run some experiments first."
            )
        return False


def find_best_model():
    """Find the best model based on evaluation metrics from MLflow"""
    try:
        # Set tracking URI to the SQLite backend
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        if not experiments:
            print("No experiments found in MLflow")
            return None

        # Get the latest experiment
        experiment_id = experiments[0].experiment_id

        # Get all runs for this experiment
        runs = client.search_runs(
            experiment_ids=[experiment_id], order_by=["metrics.test_accuracy DESC"]
        )

        if not runs:
            print("No runs found in MLflow")
            return None

        # Get the best run
        best_run = runs[0]
        run_id = best_run.info.run_id
        test_accuracy = best_run.data.metrics.get("test_accuracy", 0)

        print(
            f"Found best model with run_id {run_id} and test accuracy {test_accuracy:.4f}"
        )

        # Get parameters from best run
        params = best_run.data.params
        n_estimators = int(params.get("n_estimators", 100))
        max_depth = params.get("max_depth", "None")
        if max_depth != "None":
            max_depth = int(max_depth)
        else:
            max_depth = None
        min_samples_split = int(params.get("min_samples_split", 2))

        # Construct model filename
        model_filename = (
            f"churn_model_{n_estimators}_{max_depth}_{min_samples_split}.joblib"
        )

        if os.path.exists(model_filename):
            print(f"Loading best model from {model_filename}")
            return joblib.load(model_filename)
        else:
            print(f"Model file {model_filename} not found, using default model")
            return None
    except Exception as e:
        print(f"Error finding best model: {e}")
        return None
