# pipelines/data_pipeline.py
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
import warnings
import mlflow
from pipelines.mlflow_elastic import get_es_logger

warnings.filterwarnings("ignore")


def load_and_prepare_data(train_path="churn_80.csv", test_path="churn_20.csv"):
    """Loads, cleans, and prepares data for training and evaluation."""
    print(f"Loading data from {train_path} and {test_path}")

    # Initialize MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Get Elasticsearch logger
    es_logger = get_es_logger()
    
    # Start system monitoring
    es_logger.start_system_monitoring(interval=30)  # Monitor every 30 seconds

    # Check if files exist
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Data files not found. Please ensure {train_path} and {test_path} exist."
        )

    # Load CSVs
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Log data statistics to Elasticsearch
    with mlflow.start_run(run_name="data_preparation"):
        run_id = mlflow.active_run().info.run_id
        
        # Calculate basic statistics
        data_stats = {
            "train_rows": len(df_train),
            "test_rows": len(df_test),
            "total_rows": len(df_train) + len(df_test),
            "num_features": len(df_train.columns) - 1,  # Excluding target
            "churn_rate_train": df_train["Churn"].mean() * 100,
            "churn_rate_test": df_test["Churn"].mean() * 100,
            "missing_values_train": df_train.isnull().sum().sum(),
            "missing_values_test": df_test.isnull().sum().sum()
        }
        
        # Log to MLflow
        for key, value in data_stats.items():
            mlflow.log_metric(key, value)
        
        # Log to Elasticsearch
        es_logger.log_data_stats(run_id, data_stats)

    # Fill missing numeric values with the column mean
    for col in df_train.select_dtypes(include=["float64", "int64"]).columns:
        df_train[col].fillna(df_train[col].mean(), inplace=True)
        df_test[col].fillna(df_test[col].mean(), inplace=True)

    # Encode categorical features using OrdinalEncoder
    categorical_features = ["International plan", "Voice mail plan"]
    encoder = OrdinalEncoder()
    df_train[categorical_features] = encoder.fit_transform(
        df_train[categorical_features]
    )
    df_test[categorical_features] = encoder.transform(df_test[categorical_features])

    # One-hot encode the "State" column
    df_train = pd.get_dummies(df_train, columns=["State"], prefix="State")
    df_test = pd.get_dummies(df_test, columns=["State"], prefix="State")

    # Align test set columns with training set
    df_test = df_test.reindex(columns=df_train.columns, fill_value=0)

    # Normalize data using MinMaxScaler
    scaler = MinMaxScaler()
    df_train_scaled = pd.DataFrame(
        scaler.fit_transform(df_train), columns=df_train.columns
    )
    df_test_scaled = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns)

    # Drop redundant features (if they exist)
    redundant_features = [
        "Total day charge",
        "Total eve charge",
        "Total night charge",
        "Total intl charge",
    ]
    df_train_scaled.drop(columns=redundant_features, inplace=True, errors="ignore")
    df_test_scaled.drop(columns=redundant_features, inplace=True, errors="ignore")

    # Separate features and labels
    X_train = df_train_scaled.drop(columns=["Churn"])
    y_train = df_train_scaled["Churn"]
    X_test = df_test_scaled.drop(columns=["Churn"])
    y_test = df_test_scaled["Churn"]

    print(
        f"Data prepared: X_train shape: {X_train.shape}, X_test shape: {X_test.shape}"
    )

    # Save the processed data as a joblib file
    processed_data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }
    joblib.dump(processed_data, "processed_data.joblib")
    print("✅ Processed data saved as 'processed_data.joblib'")

    # Create directory if it doesn't exist
    os.makedirs("data/processed", exist_ok=True)

    # Save processed data
    X_train.to_csv("data/processed/X_train.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    load_and_prepare_data()
