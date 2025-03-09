# pipelines/mlflow_elastic.py
import os
import mlflow
from elasticsearch import Elasticsearch
from datetime import datetime
import json
import logging
import psutil
import platform
import time
import threading

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
        self.system_monitor_thread = None
        self.stop_monitoring = False
        
        try:
            # Use Elasticsearch client with compatibility mode for version 7.x
            self.es = Elasticsearch([es_host], timeout=30, retry_on_timeout=True)
            logger.info(f"Connected to Elasticsearch at {es_host}")
            
            # Create indices if they don't exist
            indices = [
                es_index,
                f"{es_index}-performance",
                f"{es_index}-system",
                f"{es_index}-data"
            ]
            
            for index in indices:
                if not self.es.indices.exists(index=index):
                    self.es.indices.create(index=index)
                    logger.info(f"Created index {index}")
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
                self.es.index(index=self.es_index, body=document)
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
                self.es.index(index=self.es_index, body=document)
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
        }
        # Add metrics to document
        for k, v in metrics.items():
            document[k] = v
        
        try:
            self.es.index(index=f"{self.es_index}-performance", body=document)
            logger.info(f"Logged model performance to Elasticsearch")
        except Exception as e:
            logger.error(f"Failed to log model performance to Elasticsearch: {str(e)}")
    
    def log_data_stats(self, run_id, data_stats):
        """
        Log data statistics to Elasticsearch
        
        Args:
            run_id: MLflow run ID
            data_stats: Dictionary of data statistics
        """
        if not self.es:
            logger.warning("Elasticsearch connection not available")
            return
            
        timestamp = datetime.now().isoformat()
        
        document = {
            "run_id": run_id,
            "timestamp": timestamp,
            **data_stats
        }
        
        try:
            self.es.index(index=f"{self.es_index}-data", body=document)
            logger.info(f"Logged data statistics to Elasticsearch")
        except Exception as e:
            logger.error(f"Failed to log data statistics to Elasticsearch: {str(e)}")
    
    def start_system_monitoring(self, interval=10):
        """
        Start a background thread to monitor system resources
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.system_monitor_thread and self.system_monitor_thread.is_alive():
            logger.warning("System monitoring is already running")
            return
            
        self.stop_monitoring = False
        self.system_monitor_thread = threading.Thread(
            target=self._monitor_system,
            args=(interval,),
            daemon=True
        )
        self.system_monitor_thread.start()
        logger.info(f"Started system monitoring with interval {interval}s")
    
    def stop_system_monitoring(self):
        """Stop the system monitoring thread"""
        if self.system_monitor_thread and self.system_monitor_thread.is_alive():
            self.stop_monitoring = True
            self.system_monitor_thread.join(timeout=5)
            logger.info("Stopped system monitoring")
    
    def _monitor_system(self, interval):
        """
        Monitor system resources periodically
        
        Args:
            interval: Monitoring interval in seconds
        """
        if not self.es:
            logger.warning("Elasticsearch connection not available")
            return
            
        while not self.stop_monitoring:
            timestamp = datetime.now().isoformat()
            
            # Collect system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            document = {
                "timestamp": timestamp,
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024 ** 3),
                "memory_total_gb": memory.total / (1024 ** 3),
                "disk_percent": disk.percent,
                "disk_used_gb": disk.used / (1024 ** 3),
                "disk_total_gb": disk.total / (1024 ** 3),
                "system": platform.system(),
                "node": platform.node()
            }
            
            try:
                self.es.index(index=f"{self.es_index}-system", body=document)
                logger.debug(f"Logged system metrics: CPU {cpu_percent}%, Memory {memory.percent}%")
            except Exception as e:
                logger.error(f"Failed to log system metrics to Elasticsearch: {str(e)}")
            
            time.sleep(interval)

# Helper function to get current run ID
def get_current_run_id():
    return mlflow.active_run().info.run_id if mlflow.active_run() else None

# Singleton pattern for the logger
_es_logger = None

def get_es_logger(es_host="http://localhost:9200", es_index="mlflow-metrics"):
    """
    Get or create a singleton MLflowElasticLogger instance
    
    Args:
        es_host: Elasticsearch host URL
        es_index: Elasticsearch index name
    
    Returns:
        MLflowElasticLogger: Singleton instance
    """
    global _es_logger
    if _es_logger is None:
        _es_logger = MLflowElasticLogger(es_host, es_index)
    return _es_logger
