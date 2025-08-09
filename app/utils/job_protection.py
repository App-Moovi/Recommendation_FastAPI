import os
import time
import logging
import threading
import tempfile
from functools import wraps
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class JobLockManager:
    """In-memory job lock manager with file-based persistence fallback"""
    
    def __init__(self, use_file_locks: bool = True):
        self._locks: Dict[str, Dict] = {}
        self._lock = threading.RLock()
        self.use_file_locks = use_file_locks
        self.lock_dir = Path(tempfile.gettempdir()) / "job_locks"
        
        if self.use_file_locks:
            self.lock_dir.mkdir(exist_ok=True)
    
    def acquire_lock(self, job_name: str, timeout: int) -> bool:
        """Acquire a lock for the given job"""
        with self._lock:
            lock_key = f"job_lock:{job_name}"
            current_time = time.time()
            
            # Clean up expired locks first
            self._cleanup_expired_locks()
            
            # Check if lock already exists and is not expired
            if lock_key in self._locks:
                lock_info = self._locks[lock_key]
                if current_time < lock_info['expires_at']:
                    return False  # Lock still active
                else:
                    # Lock expired, remove it
                    self._remove_lock(lock_key)
            
            # Try file-based lock as additional safety
            if self.use_file_locks and not self._acquire_file_lock(job_name, timeout):
                return False
            
            # Acquire the lock
            self._locks[lock_key] = {
                'acquired_at': current_time,
                'expires_at': current_time + timeout,
                'thread_id': threading.get_ident(),
                'process_id': os.getpid()
            }
            return True
    
    def release_lock(self, job_name: str) -> bool:
        """Release a lock for the given job"""
        with self._lock:
            lock_key = f"job_lock:{job_name}"
            
            if lock_key in self._locks:
                del self._locks[lock_key]
                
                # Remove file lock if using file-based locking
                if self.use_file_locks:
                    self._release_file_lock(job_name)
                
                return True
            return False
    
    def is_locked(self, job_name: str) -> bool:
        """Check if a job is currently locked"""
        with self._lock:
            lock_key = f"job_lock:{job_name}"
            current_time = time.time()
            
            if lock_key in self._locks:
                lock_info = self._locks[lock_key]
                if current_time < lock_info['expires_at']:
                    return True
                else:
                    # Lock expired, remove it
                    self._remove_lock(lock_key)
            
            # Also check file-based lock
            if self.use_file_locks:
                return self._check_file_lock(job_name)
            
            return False
    
    def force_unlock(self, job_name: str) -> bool:
        """Force unlock a job (use with caution)"""
        with self._lock:
            lock_key = f"job_lock:{job_name}"
            
            removed = False
            if lock_key in self._locks:
                del self._locks[lock_key]
                removed = True
            
            if self.use_file_locks:
                if self._release_file_lock(job_name):
                    removed = True
            
            return removed
    
    def _cleanup_expired_locks(self):
        """Clean up expired locks from memory"""
        current_time = time.time()
        expired_keys = [
            key for key, lock_info in self._locks.items()
            if current_time >= lock_info['expires_at']
        ]
        
        for key in expired_keys:
            self._remove_lock(key)
    
    def _remove_lock(self, lock_key: str):
        """Remove a lock and its associated file lock"""
        if lock_key in self._locks:
            del self._locks[lock_key]
        
        if self.use_file_locks:
            job_name = lock_key.replace("job_lock:", "")
            self._release_file_lock(job_name)
    
    def _acquire_file_lock(self, job_name: str, timeout: int) -> bool:
        """Acquire a file-based lock"""
        try:
            lock_file = self.lock_dir / f"{job_name}.lock"
            current_time = time.time()
            
            # Check if lock file exists and is not expired
            if lock_file.exists():
                try:
                    with open(lock_file, 'r') as f:
                        lock_data = f.read().strip().split('\n')
                        if len(lock_data) >= 2:
                            expires_at = float(lock_data[1])
                            if current_time < expires_at:
                                return False  # Lock still active
                except (ValueError, IOError):
                    # Invalid lock file, remove it
                    lock_file.unlink(missing_ok=True)
            
            # Create new lock file
            with open(lock_file, 'w') as f:
                f.write(f"{os.getpid()}\n")
                f.write(f"{current_time + timeout}\n")
                f.write(f"{threading.get_ident()}\n")
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to acquire file lock for {job_name}: {e}")
            return True  # Fallback to memory-only locking
    
    def _release_file_lock(self, job_name: str) -> bool:
        """Release a file-based lock"""
        try:
            lock_file = self.lock_dir / f"{job_name}.lock"
            if lock_file.exists():
                lock_file.unlink()
                return True
        except Exception as e:
            logger.warning(f"Failed to release file lock for {job_name}: {e}")
        return False
    
    def _check_file_lock(self, job_name: str) -> bool:
        """Check if file-based lock exists and is active"""
        try:
            lock_file = self.lock_dir / f"{job_name}.lock"
            if not lock_file.exists():
                return False
            
            current_time = time.time()
            with open(lock_file, 'r') as f:
                lock_data = f.read().strip().split('\n')
                if len(lock_data) >= 2:
                    expires_at = float(lock_data[1])
                    return current_time < expires_at
            
        except Exception as e:
            logger.warning(f"Failed to check file lock for {job_name}: {e}")
        
        return False

# Global instance
lock_manager = JobLockManager()

def prevent_overlap(job_name: str, timeout: int = 3600):
    """
    Decorator to prevent job overlap using in-memory locks with file-based fallback
    
    Args:
        job_name: Unique name for the job
        timeout: Maximum time in seconds the job can run before lock expires
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Try to acquire lock
            if lock_manager.acquire_lock(job_name, timeout):
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
                    lock_manager.release_lock(job_name)
                    logger.info(f"Released lock for job {job_name}")
            else:
                logger.warning(f"Job {job_name} is already running, skipping execution")
                return None
        return wrapper
    return decorator

def is_job_running(job_name: str) -> bool:
    """Check if a job is currently running"""
    return lock_manager.is_locked(job_name)

def force_unlock_job(job_name: str) -> bool:
    """Force unlock a job (use with caution)"""
    result = lock_manager.force_unlock(job_name)
    if result:
        logger.warning(f"Force unlocked job {job_name}")
    return result

# Alternative simpler implementation using only in-memory locks
class SimpleJobLockManager:
    """Simple in-memory only job lock manager"""
    
    def __init__(self):
        self._locks: Dict[str, float] = {}
        self._lock = threading.RLock()
    
    def acquire_lock(self, job_name: str, timeout: int) -> bool:
        """Acquire a lock for the given job"""
        with self._lock:
            logger.info(f"Attempting to acquire lock for job {job_name} with timeout {timeout} seconds. Locks: {self._locks} . LockManager: {self}")
            current_time = time.time()
            
            # Check if lock exists and is not expired
            if job_name in self._locks:
                if current_time < self._locks[job_name]:
                    return False  # Lock still active
                else:
                    # Lock expired, remove it
                    del self._locks[job_name]
            
            # Acquire the lock
            self._locks[job_name] = current_time + timeout
            return True
    
    def release_lock(self, job_name: str) -> bool:
        logger.info(f"Attempting to release lock for job {job_name} with timeout {timeout} seconds. Locks: {self._locks} . LockManager: {self}")
        """Release a lock for the given job"""
        with self._lock:
            return self._locks.pop(job_name, None) is not None
    
    def is_locked(self, job_name: str) -> bool:
        """Check if a job is currently locked"""
        logger.info(f"Attempting to check is locked for job {job_name} with timeout {timeout} seconds. Locks: {self._locks} . LockManager: {self}")

        with self._lock:
            current_time = time.time()
            
            if job_name in self._locks:
                if current_time < self._locks[job_name]:
                    return True
                else:
                    # Lock expired, remove it
                    del self._locks[job_name]
            
            return False
    
    def force_unlock(self, job_name: str) -> bool:
        """Force unlock a job"""
        logger.info(f"Attempting to force unlock job {job_name} with timeout {timeout} seconds. Locks: {self._locks} . LockManager: {self}")
        with self._lock:
            return self._locks.pop(job_name, None) is not None

# Uncomment to use the simpler version instead:
lock_manager = SimpleJobLockManager()
logger.info(lock_manager)