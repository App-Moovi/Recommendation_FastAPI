import redis
import time
import logging
from functools import wraps
from app.config import settings

logger = logging.getLogger(__name__)

# Initialize Redis client with settings
try:
    redis_client = redis.Redis(
        host=settings.REDIS_HOST if hasattr(settings, 'REDIS_HOST') else 'localhost',
        port=settings.REDIS_PORT if hasattr(settings, 'REDIS_PORT') else 6379,
        db=settings.REDIS_DB if hasattr(settings, 'REDIS_DB') else 0,
        decode_responses=True
    )
    # Test connection
    redis_client.ping()
    logger.info("Redis connection established for job protection")
except Exception as e:
    logger.warning(f"Redis not available for job protection: {e}")
    redis_client = None

def prevent_overlap(job_name: str, timeout: int = 3600):
    """
    Decorator to prevent job overlap using Redis locks
    
    Args:
        job_name: Unique name for the job
        timeout: Maximum time in seconds the job can run before lock expires
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not redis_client:
                logger.warning(f"Redis not available, running {job_name} without overlap protection")
                return func(*args, **kwargs)
            
            lock_key = f"job_lock:{job_name}"
            start_time = time.time()
            
            # Try to acquire lock
            if redis_client.set(lock_key, "locked", nx=True, ex=timeout):
                logger.info(f"Acquired lock for job {job_name}")
                try:
                    result = func(*args, **kwargs)
                    elapsed = time.time() - start_time
                    logger.info(f"Job {job_name} completed in {elapsed:.2f} seconds")
                    return result
                except Exception as e:
                    logger.error(f"Error in job {job_name}: {e}")
                    raise
                finally:
                    redis_client.delete(lock_key)
                    logger.info(f"Released lock for job {job_name}")
            else:
                logger.warning(f"Job {job_name} is already running, skipping execution")
                return None
        return wrapper
    return decorator

def is_job_running(job_name: str) -> bool:
    """Check if a job is currently running"""
    if not redis_client:
        return False
    
    lock_key = f"job_lock:{job_name}"
    return redis_client.exists(lock_key) > 0

def force_unlock_job(job_name: str) -> bool:
    """Force unlock a job (use with caution)"""
    if not redis_client:
        return False
    
    lock_key = f"job_lock:{job_name}"
    result = redis_client.delete(lock_key)
    if result:
        logger.warning(f"Force unlocked job {job_name}")
    return bool(result)