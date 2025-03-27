import logging
import os
from typing import Dict, Any, Callable, List, Optional
import time
import threading
from concurrent.futures import Future

# Import distributed computing libraries
try:
    import dask
    import dask.distributed as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

logger = logging.getLogger(__name__)

class DistributedCompute:
    """
    Manages distributed computing resources using Dask or Ray
    to execute tasks across available computational resources
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the distributed computing system
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self.scheduler_address = config.get("scheduler_address")
        self.worker_threads = config.get("worker_threads", os.cpu_count())
        self.memory_limit = config.get("memory_limit", "4GB")
        
        self.client = None
        self.ray_initialized = False
        self.engine = "none"
        
        if not self.enabled:
            logger.info("Distributed computing is disabled")
            return
        
        # Try to initialize distributed computing
        self._initialize()
        
        # Start monitoring thread if distributed computing is enabled
        if self.engine != "none":
            self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
            self.monitor_thread.start()
    
    def _initialize(self) -> None:
        """Initialize the distributed computing engine (Dask or Ray)"""
        # Try to initialize Dask first
        if DASK_AVAILABLE:
            try:
                if self.scheduler_address:
                    # Connect to existing Dask scheduler
                    self.client = dd.Client(self.scheduler_address)
                    logger.info(f"Connected to Dask scheduler at {self.scheduler_address}")
                else:
                    # Start a local Dask cluster
                    self.client = dd.Client(
                        n_workers=self.worker_threads,
                        threads_per_worker=1,
                        memory_limit=self.memory_limit
                    )
                    logger.info(f"Started local Dask cluster with {self.worker_threads} workers")
                
                self.engine = "dask"
                return
            except Exception as e:
                logger.warning(f"Failed to initialize Dask: {str(e)}")
        
        # If Dask fails or is not available, try Ray
        if RAY_AVAILABLE:
            try:
                # Initialize Ray
                if not ray.is_initialized():
                    ray.init(
                        ignore_reinit_error=True,
                        num_cpus=self.worker_threads,
                        _memory=self.memory_limit
                    )
                    self.ray_initialized = True
                    logger.info(f"Initialized Ray with {self.worker_threads} CPUs")
                else:
                    self.ray_initialized = True
                    logger.info("Connected to existing Ray runtime")
                
                self.engine = "ray"
                return
            except Exception as e:
                logger.warning(f"Failed to initialize Ray: {str(e)}")
        
        # If both fail, fall back to local execution
        logger.warning("Distributed computing not available, falling back to local execution")
        self.engine = "none"
    
    def submit(self, func: Callable, *args, **kwargs) -> Future:
        """
        Submit a task to be executed in the distributed environment
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Future object representing the computation
        """
        if not self.enabled or self.engine == "none":
            # Execute locally using a simple Future
            future = LocalFuture()
            try:
                result = func(*args, **kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            return future
        
        # Execute using Dask
        if self.engine == "dask":
            return self.client.submit(func, *args, **kwargs)
        
        # Execute using Ray
        if self.engine == "ray":
            @ray.remote
            def ray_wrapper(f, *a, **k):
                return f(*a, **k)
            
            ray_future = ray_wrapper.remote(func, *args, **kwargs)
            return RayFutureWrapper(ray_future)
    
    def map(self, func: Callable, items: List[Any], **kwargs) -> List[Future]:
        """
        Map a function over a collection of items
        
        Args:
            func: Function to apply
            items: Collection of items
            **kwargs: Additional keyword arguments for the function
            
        Returns:
            List of Future objects
        """
        return [self.submit(func, item, **kwargs) for item in items]
    
    def gather(self, futures: List[Future]) -> List[Any]:
        """
        Gather results from multiple futures
        
        Args:
            futures: List of Future objects
            
        Returns:
            List of results
        """
        if not self.enabled or self.engine == "none":
            return [future.result() for future in futures]
        
        if self.engine == "dask":
            return self.client.gather(futures)
        
        if self.engine == "ray":
            return [future.result() for future in futures]
    
    def _monitor_resources(self) -> None:
        """Monitor resource usage in the distributed system"""
        while self.enabled and self.engine != "none":
            try:
                if self.engine == "dask" and self.client:
                    workers = self.client.scheduler_info()["workers"]
                    total_memory = sum(w["memory_limit"] for w in workers.values())
                    used_memory = sum(w["memory"] for w in workers.values())
                    usage_pct = (used_memory / total_memory) * 100 if total_memory > 0 else 0
                    
                    logger.debug(f"Dask memory usage: {usage_pct:.1f}% ({used_memory / 1e9:.2f}GB / {total_memory / 1e9:.2f}GB)")
                    
                    # Log warning if memory usage is high
                    if usage_pct > 80:
                        logger.warning(f"High memory usage in Dask cluster: {usage_pct:.1f}%")
                
                elif self.engine == "ray" and self.ray_initialized:
                    resources = ray.available_resources()
                    total_cpus = ray.cluster_resources().get("CPU", 0)
                    available_cpus = resources.get("CPU", 0)
                    used_cpus = total_cpus - available_cpus
                    usage_pct = (used_cpus / total_cpus) * 100 if total_cpus > 0 else 0
                    
                    logger.debug(f"Ray CPU usage: {usage_pct:.1f}% ({used_cpus}/{total_cpus})")
                    
                    # Log warning if CPU usage is high
                    if usage_pct > 80:
                        logger.warning(f"High CPU usage in Ray: {usage_pct:.1f}%")
            
            except Exception as e:
                logger.error(f"Error monitoring distributed resources: {str(e)}")
            
            # Sleep for a few seconds before checking again
            time.sleep(30)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the distributed computing system
        
        Returns:
            Dictionary with status information
        """
        status = {
            "enabled": self.enabled,
            "engine": self.engine,
            "workers": self.worker_threads
        }
        
        if self.engine == "dask" and self.client:
            try:
                workers_info = self.client.scheduler_info()["workers"]
                status.update({
                    "connected_workers": len(workers_info),
                    "total_memory": sum(w["memory_limit"] for w in workers_info.values()),
                    "used_memory": sum(w["memory"] for w in workers_info.values())
                })
            except Exception as e:
                status["error"] = str(e)
        
        elif self.engine == "ray" and self.ray_initialized:
            try:
                status.update({
                    "total_cpus": ray.cluster_resources().get("CPU", 0),
                    "available_cpus": ray.available_resources().get("CPU", 0),
                    "total_memory": ray.cluster_resources().get("memory", 0),
                    "available_memory": ray.available_resources().get("memory", 0)
                })
            except Exception as e:
                status["error"] = str(e)
        
        return status
    
    def shutdown(self) -> None:
        """Shutdown the distributed computing system"""
        if self.engine == "dask" and self.client:
            try:
                self.client.close()
                logger.info("Dask client closed")
            except Exception as e:
                logger.error(f"Error closing Dask client: {str(e)}")
        
        elif self.engine == "ray" and self.ray_initialized:
            try:
                if ray.is_initialized():
                    ray.shutdown()
                    logger.info("Ray shutdown")
            except Exception as e:
                logger.error(f"Error shutting down Ray: {str(e)}")
        
        self.enabled = False
        self.engine = "none"


class LocalFuture(Future):
    """Simple Future implementation for local execution"""
    pass


class RayFutureWrapper:
    """Wrapper for Ray ObjectRef to make it compatible with Future interface"""
    
    def __init__(self, ray_future):
        self.ray_future = ray_future
    
    def result(self):
        """Get the result of the future"""
        return ray.get(self.ray_future)
    
    def done(self):
        """Check if the future is done"""
        return ray.wait([self.ray_future], timeout=0)[0] != []
    
    def cancel(self):
        """Cancel the future (not supported in Ray)"""
        logger.warning("Cancel not supported for Ray futures")
        return False
