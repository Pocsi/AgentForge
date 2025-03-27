import logging
from typing import Dict, Any, List, Optional
import psutil
import platform
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class EdgeAnalyzer:
    """Edge computing resource analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_nodes = config.get("max_nodes", 10)
        self.min_performance = config.get("min_performance", 0.7)
        self.resource_types = config.get("resource_types", ["cpu", "gpu", "memory", "network"])
        
        # Initialize node registry
        self.nodes = {}
    
    async def get_available_resources(self) -> List[Dict[str, Any]]:
        """Get available edge computing resources"""
        try:
            # Get system information
            system_info = self._get_system_info()
            
            # Get resource metrics
            metrics = self._get_resource_metrics()
            
            # Create resource description
            resources = []
            
            # CPU resources
            if "cpu" in self.resource_types:
                cpu_resources = self._analyze_cpu_resources(system_info, metrics)
                resources.extend(cpu_resources)
            
            # GPU resources
            if "gpu" in self.resource_types:
                gpu_resources = self._analyze_gpu_resources(system_info, metrics)
                resources.extend(gpu_resources)
            
            # Memory resources
            if "memory" in self.resource_types:
                memory_resources = self._analyze_memory_resources(system_info, metrics)
                resources.extend(memory_resources)
            
            # Network resources
            if "network" in self.resource_types:
                network_resources = self._analyze_network_resources(system_info, metrics)
                resources.extend(network_resources)
            
            # Filter by performance threshold
            resources = [
                r for r in resources
                if r.get("performance", 0) >= self.min_performance
            ]
            
            # Sort by performance
            resources.sort(key=lambda x: x.get("performance", 0), reverse=True)
            
            # Limit to max nodes
            resources = resources[:self.max_nodes]
            
            return resources
            
        except Exception as e:
            logger.error(f"Error getting available resources: {str(e)}")
            return []
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            return {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "machine": platform.machine(),
                "node": platform.node(),
                "system": platform.system(),
                "version": platform.version()
            }
            
        except Exception as e:
            logger.error(f"Error getting system info: {str(e)}")
            return {}
    
    def _get_resource_metrics(self) -> Dict[str, Any]:
        """Get current resource metrics"""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "network_io": psutil.net_io_counters()._asdict()
            }
            
        except Exception as e:
            logger.error(f"Error getting resource metrics: {str(e)}")
            return {}
    
    def _analyze_cpu_resources(
        self,
        system_info: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Analyze CPU resources"""
        try:
            cpu_count = psutil.cpu_count()
            cpu_percent = metrics.get("cpu_percent", 0)
            
            # Calculate performance score
            performance = 1 - (cpu_percent / 100)
            
            resources = []
            for i in range(cpu_count):
                resource = {
                    "id": f"cpu_{i}",
                    "type": "cpu",
                    "cores": 1,
                    "performance": performance,
                    "utilization": cpu_percent,
                    "capabilities": ["compute", "parallel_processing"],
                    "status": "active"
                }
                resources.append(resource)
            
            return resources
            
        except Exception as e:
            logger.error(f"Error analyzing CPU resources: {str(e)}")
            return []
    
    def _analyze_gpu_resources(
        self,
        system_info: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Analyze GPU resources"""
        try:
            # In a real implementation, this would use CUDA or other GPU libraries
            # For now, we'll create a placeholder
            resources = []
            
            # Check for NVIDIA GPU
            if "nvidia" in system_info.get("processor", "").lower():
                resource = {
                    "id": "gpu_0",
                    "type": "gpu",
                    "model": "NVIDIA GPU",
                    "performance": 0.9,
                    "capabilities": ["compute", "graphics", "machine_learning"],
                    "status": "active"
                }
                resources.append(resource)
            
            return resources
            
        except Exception as e:
            logger.error(f"Error analyzing GPU resources: {str(e)}")
            return []
    
    def _analyze_memory_resources(
        self,
        system_info: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Analyze memory resources"""
        try:
            memory = psutil.virtual_memory()
            memory_percent = metrics.get("memory_percent", 0)
            
            # Calculate performance score
            performance = 1 - (memory_percent / 100)
            
            resource = {
                "id": "memory_0",
                "type": "memory",
                "total": memory.total,
                "available": memory.available,
                "performance": performance,
                "utilization": memory_percent,
                "capabilities": ["storage", "caching"],
                "status": "active"
            }
            
            return [resource]
            
        except Exception as e:
            logger.error(f"Error analyzing memory resources: {str(e)}")
            return []
    
    def _analyze_network_resources(
        self,
        system_info: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Analyze network resources"""
        try:
            network_io = metrics.get("network_io", {})
            
            # Calculate performance score based on bandwidth
            total_bytes = network_io.get("bytes_sent", 0) + network_io.get("bytes_recv", 0)
            performance = min(1.0, total_bytes / (1024 * 1024 * 1024))  # Normalize to GB
            
            resource = {
                "id": "network_0",
                "type": "network",
                "bytes_sent": network_io.get("bytes_sent", 0),
                "bytes_recv": network_io.get("bytes_recv", 0),
                "performance": performance,
                "capabilities": ["communication", "data_transfer"],
                "status": "active"
            }
            
            return [resource]
            
        except Exception as e:
            logger.error(f"Error analyzing network resources: {str(e)}")
            return []
    
    async def get_node_metrics(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Get performance metrics for a specific node"""
        try:
            node_id = node.get("id")
            node_type = node.get("type")
            
            if node_id not in self.nodes:
                self.nodes[node_id] = {
                    "id": node_id,
                    "type": node_type,
                    "metrics_history": [],
                    "last_update": datetime.now().isoformat()
                }
            
            # Get current metrics
            metrics = self._get_resource_metrics()
            
            # Update node metrics
            self.nodes[node_id]["metrics_history"].append({
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics
            })
            
            # Keep only last 100 metrics
            self.nodes[node_id]["metrics_history"] = self.nodes[node_id]["metrics_history"][-100:]
            self.nodes[node_id]["last_update"] = datetime.now().isoformat()
            
            return {
                "node_id": node_id,
                "type": node_type,
                "current_metrics": metrics,
                "metrics_history": self.nodes[node_id]["metrics_history"],
                "last_update": self.nodes[node_id]["last_update"]
            }
            
        except Exception as e:
            logger.error(f"Error getting node metrics: {str(e)}")
            return {} 