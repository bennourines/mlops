import os
import mlflow
from elasticsearch import Elasticsearch
from datetime import datetime
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLflowElasticLogger:
    """
    A class to log MLflow metrics and parameters to Elasticsearch
    """
    
    def __init__(self, es_host="http://localhost:9200", es_index="mlflow-metrics"):
        """
        Initialize the logger
        
        Args:
            es_host: Elasticsearch host URL
            es_index: Elasticsearch index name
        """
        self.es_host = es_host
        self.es_index = es_index
        self.es = None
        
        try:
            self.es = Elasticsearch([es_host])
            logger.info(f"Connected to Elasticsearch at {es_host}")
            
            # Create index if it doesn't exist
            if not self.es.indices.exists(index=es_index):
                self.es.indices.create(index=es_index)
                logger.info(f"Created index {es_index}")
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {str(e)}")
    
    def log_metrics(self, run_id, metrics, step=None):
        """
        Log metrics to Elasticsearch
        
        Args:
            run_id: MLflow run ID
            metrics: Dictionary of metrics
            step: Step number (optional)
        """
        if not self.es:
            logger.warning("Elasticsearch connection not available")
            return
            
        timestamp = datetime.now().isoformat()
        
        for metric_name, metric_value in metrics.items():
            document = {
                "run_id": run_id,
                "timestamp": timestamp,
                "metric_name": metric_name,
                "metric_value": float(metric_value),
                "step": step
            }
            
            try:
                self.es.index(index=self.es_index, document=document)
                logger.debug(f"Logged metric {metric_name}={metric_value} to Elasticsearch")
            except Exception as e:
                logger.error(f"Failed to log metric to Elasticsearch: {str(e)}")
    
    def log_parameters(self, run_id, parameters):
        """
        Log parameters to Elasticsearch
        
        Args:
            run_id: MLflow run ID
            parameters: Dictionary of parameters
        """
        if not self.es:
            logger.warning("Elasticsearch connection not available")
            return
            
        timestamp = datetime.now().isoformat()
        
        for param_name, param_value in parameters.items():
            document = {
                "run_id": run_id,
                "timestamp": timestamp,
                "param_name": param_name,
                "param_value": str(param_value),
            }
            
            try:
                self.es.index(index=self.es_index, document=document)
                logger.debug(f"Logged parameter {param_name}={param_value} to Elasticsearch")
            except Exception as e:
                logger.error(f"Failed to log parameter to Elasticsearch: {str(e)}")
    
    def log_model_performance(self, run_id, model_name, metrics):
        """
        Log model performance to Elasticsearch
        
        Args:
            run_id: MLflow run ID
            model_name: Name of the model
            metrics: Dictionary of performance metrics
        """
        if not self.es:
            logger.warning("Elasticsearch connection not available")
            return
            
        timestamp = datetime.now().isoformat()
        
        document = {
            "run_id": run_id,
            "timestamp": timestamp,
            "model_name": model_name,
            **metrics
        }
        
        try:
            self.es.index(index=f"{self.es_index}-performance", document=document)
            logger.info(f"Logged model performance to Elasticsearch")
        except Exception as e:
            logger.error(f"Failed to log model performance to Elasticsearch: {str(e)}")

# Helper function to get current run ID
def get_current_run_id():
    return mlflow.active_run().info.run_id if mlflow.active_run() else None
