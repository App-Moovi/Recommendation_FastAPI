import psutil
import logging
import gc
from typing import Optional, Callable
from functools import wraps
from app.utils.logger import timed

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Monitor and limit memory usage in background tasks"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_usage()
    
    @timed
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    @timed
    def get_memory_percent(self) -> float:
        """Get memory usage as percentage of total system memory"""
        return self.process.memory_percent()
    
    @timed
    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        return self.process.cpu_percent(interval=0.1)
    
    @timed
    def check_memory_usage(self, max_memory_mb: float = 1000) -> bool:
        """
        Check if memory usage is within limits
        
        Args:
            max_memory_mb: Maximum allowed memory in MB
            
        Returns:
            bool: True if within limits, False if exceeded
        """
        current_memory = self.get_memory_usage()
        
        if current_memory > max_memory_mb:
            logger.warning(
                f"Memory usage {current_memory:.1f}MB exceeds limit {max_memory_mb}MB "
                f"({self.get_memory_percent():.1f}% of system memory)"
            )
            return False
        
        return True
    
    @timed
    def log_memory_usage(self, task_name: str = "", include_delta: bool = True):
        """
        Log current memory usage
        
        Args:
            task_name: Name of the task for logging
            include_delta: Whether to include memory change since initialization
        """
        current_memory = self.get_memory_usage()
        memory_percent = self.get_memory_percent()
        cpu_percent = self.get_cpu_usage()
        
        message = f"{task_name} - Memory: {current_memory:.1f}MB ({memory_percent:.1f}%), CPU: {cpu_percent:.1f}%"
        
        if include_delta:
            delta = current_memory - self.initial_memory
            message += f", Delta: {delta:+.1f}MB"
        
        logger.info(message)
    
    @timed
    def force_garbage_collection(self):
        """Force garbage collection and log results"""
        before_memory = self.get_memory_usage()
        
        # Force full garbage collection
        collected = gc.collect(2)  # Collect all generations
        
        after_memory = self.get_memory_usage()
        freed = before_memory - after_memory
        
        logger.info(
            f"Garbage collection: collected {collected} objects, "
            f"freed {freed:.1f}MB (from {before_memory:.1f}MB to {after_memory:.1f}MB)"
        )
    
    @staticmethod
    @timed
    def get_system_memory_info() -> dict:
        """Get system memory information"""
        memory = psutil.virtual_memory()
        return {
            'total_mb': memory.total / 1024 / 1024,
            'available_mb': memory.available / 1024 / 1024,
            'used_mb': memory.used / 1024 / 1024,
            'percent': memory.percent
        }

def memory_limit(max_memory_mb: float = 1000, check_interval: int = 100):
    """
    Decorator to enforce memory limits on functions
    
    Args:
        max_memory_mb: Maximum allowed memory usage in MB
        check_interval: How often to check memory (every N iterations if function yields)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = MemoryMonitor()
            monitor.log_memory_usage(f"Starting {func.__name__}")
            
            # Check if function is a generator
            result = func(*args, **kwargs)
            
            if hasattr(result, '__iter__') and hasattr(result, '__next__'):
                # It's a generator, monitor during iteration
                def monitored_generator():
                    iteration = 0
                    for item in result:
                        iteration += 1
                        
                        # Check memory at intervals
                        if iteration % check_interval == 0:
                            if not monitor.check_memory_usage(max_memory_mb):
                                logger.error(
                                    f"{func.__name__} exceeded memory limit at iteration {iteration}"
                                )
                                monitor.force_garbage_collection()
                                
                                # Re-check after garbage collection
                                if not monitor.check_memory_usage(max_memory_mb):
                                    raise MemoryError(
                                        f"{func.__name__} exceeded memory limit of {max_memory_mb}MB"
                                    )
                        
                        yield item
                    
                    monitor.log_memory_usage(f"Completed {func.__name__}")
                
                return monitored_generator()
            else:
                # Regular function, just check at the end
                monitor.log_memory_usage(f"Completed {func.__name__}")
                return result
        
        return wrapper
    return decorator

class MemoryError(Exception):
    """Raised when memory limits are exceeded"""
    pass