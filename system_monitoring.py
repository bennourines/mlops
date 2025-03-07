import psutil
import platform
import time
from datetime import datetime
from elasticsearch import Elasticsearch
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemMonitor:
    """
    A class to monitor system resources and log them to Elasticsearch
    """
    
    def __init__(self, es_host="http://localhost:9200", es_index="system-metrics"):
        """
        Initialize the monitor
        
        Args:
            es_host: Elasticsearch host URL
            es_index: Elasticsearch index name
        """
        self.es_host = es_host
        self.es_index = es_index
        self.es = None
        self.hostname = platform.node()
        
        try:
            self.es = Elasticsearch([es_host])
            logger.info(f"Connected to Elasticsearch at {es_host}")
            
            # Create index if it doesn't exist
            if not self.es.indices.exists(index=es_index):
                self.es.indices.create(index=es_index)
                logger.info(f"Created index {es_index}")
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {str(e)}")
    
    def get_system_metrics(self):
        """
        Collect system metrics
        
        Returns:
            dict: Dictionary of system metrics
        """
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_total = memory.total / (1024 * 1024 * 1024)  # GB
        memory_used = memory.used / (1024 * 1024 * 1024)  # GB
        memory_percent = memory.percent
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_total = disk.total / (1024 * 1024 * 1024)  # GB
        disk_used = disk.used / (1024 * 1024 * 1024)  # GB
        disk_percent = disk.percent
        
        # Network metrics
        net_io = psutil.net_io_counters()
        net_sent = net_io.bytes_sent / (1024 * 1024)  # MB
        net_recv = net_io.bytes_recv / (1024 * 1024)  # MB
        
        # Docker metrics (if Docker is available)
        docker_containers = 0
        docker_running = 0
        try:
            import docker
            client = docker.from_env()
            all_containers = client.containers.list(all=True)
            docker_containers = len(all_containers)
            docker_running = len(client.containers.list())
        except:
            pass
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "hostname": self.hostname,
            "cpu_percent": cpu_percent,
            "cpu_count": cpu_count,
            "memory_total_gb": round(memory_total, 2),
            "memory_used_gb": round(memory_used, 2),
            "memory_percent": memory_percent,
            "disk_total_gb": round(disk_total, 2),
            "disk_used_gb": round(disk_used, 2),
            "disk_percent": disk_percent,
            "net_sent_mb": round(net_sent, 2),
            "net_recv_mb": round(net_recv, 2),
            "docker_containers": docker_containers,
            "docker_running": docker_running
        }
        
        return metrics
    
    def log_metrics(self, metrics):
        """
        Log metrics to Elasticsearch
        
        Args:
            metrics: Dictionary of metrics
        """
        if not self.es:
            logger.warning("Elasticsearch connection not available")
            return
        
        try:
            self.es.index(index=self.es_index, document=metrics)
            logger.info("System metrics logged to Elasticsearch")
        except Exception as e:
            logger.error(f"Failed to log metrics to Elasticsearch: {str(e)}")
    
    def start_monitoring(self, interval=60):
        """
        Start monitoring system at regular intervals
        
        Args:
            interval: Time between measurements in seconds
        """
        logger.info(f"Starting system monitoring every {interval} seconds")
        
        try:
            while True:
                metrics = self.get_system_metrics()
                
                # Print metrics to console
                print(f"\n--- System Metrics at {metrics['timestamp']} ---")
                print(f"CPU: {metrics['cpu_percent']}% (of {metrics['cpu_count']} cores)")
                print(f"Memory: {metrics['memory_used_gb']:.2f}GB / {metrics['memory_total_gb']:.2f}GB ({metrics['memory_percent']}%)")
                print(f"Disk: {metrics['disk_used_gb']:.2f}GB / {metrics['disk_total_gb']:.2f}GB ({metrics['disk_percent']}%)")
                print(f"Network: Sent: {metrics['net_sent_mb']:.2f}MB, Recv: {metrics['net_recv_mb']:.2f}MB")
                print(f"Docker: {metrics['docker_running']} running of {metrics['docker_containers']} total")
                
                # Log to Elasticsearch
                self.log_metrics(metrics)
                
                # Sleep until next interval
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")


if __name__ == "__main__":
    monitor = SystemMonitor()
    monitor.start_monitoring()
