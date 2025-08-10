import threading
import time
from typing import Set, List
import logging
from datetime import datetime
from app.database import SessionLocal
from app.background.tasks import BackgroundTasks
from app.utils.logger import timed

logger = logging.getLogger(__name__)

class AsyncUserStatsManager:
    """Manages user statistics updates asynchronously using simple threading"""
    
    # Class-level storage for pending updates
    _pending_updates: Set[int] = set()
    _lock = threading.Lock()
    _worker_thread = None
    _stop_worker = False
    _is_processing = False
    
    @classmethod
    @timed
    def initialize(cls):
        """Initialize the background worker thread"""
        if cls._worker_thread is None or not cls._worker_thread.is_alive():
            cls._stop_worker = False
            cls._worker_thread = threading.Thread(
                target=cls._process_updates_worker,
                daemon=True,
                name="UserStatsWorker"
            )
            cls._worker_thread.start()
            logger.info("User stats worker thread started")
    
    @classmethod
    @timed
    def shutdown(cls):
        """Shutdown the background worker thread"""
        cls._stop_worker = True
        if cls._worker_thread and cls._worker_thread.is_alive():
            cls._worker_thread.join(timeout=5)
            logger.info("User stats worker thread stopped")
    
    @classmethod
    @timed
    def queue_user_stats_update(cls, user_id: int) -> bool:
        """
        Queue a user stats update (non-blocking)
        
        Args:
            user_id: User ID to update stats for
            
        Returns:
            bool: True if queued successfully
        """
        try:
            with cls._lock:
                cls._pending_updates.add(user_id)
            
            logger.debug(f"Queued stats update for user {user_id}")
            
            # Ensure worker thread is running
            cls.initialize()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to queue stats update for user {user_id}: {e}")
            return False
    
    @classmethod
    @timed
    def queue_multiple_users(cls, user_ids: Set[int]) -> bool:
        """
        Queue multiple users for stats update
        
        Args:
            user_ids: Set of user IDs to update
            
        Returns:
            bool: True if all queued successfully
        """
        if not user_ids:
            return True
        
        try:
            with cls._lock:
                cls._pending_updates.update(user_ids)
            
            logger.debug(f"Queued stats update for {len(user_ids)} users")
            
            # Ensure worker thread is running
            cls.initialize()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to queue batch stats update: {e}")
            return False
    
    @classmethod
    @timed
    def get_pending_updates_count(cls) -> int:
        """Get count of pending stats updates"""
        with cls._lock:
            return len(cls._pending_updates)
    
    @classmethod
    @timed
    def is_processing(cls) -> bool:
        """Check if currently processing updates"""
        return cls._is_processing
    
    @classmethod
    @timed
    def _process_updates_worker(cls):
        """Background worker that processes pending updates every 2 minutes"""
        logger.info("Stats worker thread started")
        
        while not cls._stop_worker:
            try:
                # Wait for 2 minutes or until stop signal
                for _ in range(120):  # 120 seconds = 2 minutes
                    if cls._stop_worker:
                        break
                    time.sleep(1)
                
                if cls._stop_worker:
                    break
                
                # Get pending updates
                with cls._lock:
                    if not cls._pending_updates:
                        continue
                    
                    # Take a snapshot of pending updates and clear the set
                    updates_to_process = list(cls._pending_updates)
                    cls._pending_updates.clear()
                    cls._is_processing = True
                
                logger.info(f"Processing {len(updates_to_process)} pending stats updates")
                
                # Process in batches
                batch_size = 50
                total_successful = 0
                total_failed = 0
                
                for i in range(0, len(updates_to_process), batch_size):
                    batch = updates_to_process[i:i + batch_size]
                    successful, failed = cls._process_batch(batch)
                    total_successful += successful
                    total_failed += failed
                
                logger.info(f"Stats update complete: {total_successful} successful, {total_failed} failed")
                
            except Exception as e:
                logger.error(f"Error in stats worker thread: {e}")
            finally:
                cls._is_processing = False
        
        logger.info("Stats worker thread stopped")
    
    @classmethod
    @timed
    def _process_batch(cls, user_ids: List[int]) -> tuple:
        """
        Process a batch of user stats updates
        
        Returns:
            tuple: (successful_count, failed_count)
        """
        if not user_ids:
            return 0, 0
        
        db = SessionLocal()
        successful = 0
        failed = 0
        failed_users = []
        
        try:
            for user_id in user_ids:
                try:
                    # Update user stats
                    BackgroundTasks.update_user_summary(user_id)
                    successful += 1
                    logger.debug(f"Successfully updated stats for user {user_id}")
                except Exception as e:
                    logger.error(f"Failed to update stats for user {user_id}: {e}")
                    failed += 1
                    failed_users.append(user_id)
            
            db.commit()
            
            # Re-queue failed users
            if failed_users:
                with cls._lock:
                    cls._pending_updates.update(failed_users)
                logger.info(f"Re-queued {len(failed_users)} failed users for retry")
            
            return successful, failed
            
        except Exception as e:
            logger.error(f"Error processing user stats batch: {e}")
            db.rollback()
            
            # Re-queue all users from this batch
            with cls._lock:
                cls._pending_updates.update(user_ids)
            
            return 0, len(user_ids)
        finally:
            db.close()
    
    @classmethod
    @timed
    def force_process_now(cls) -> dict:
        """
        Force immediate processing of pending updates (for admin use)
        
        Returns:
            dict: Processing results
        """
        if cls._is_processing:
            return {
                "status": "already_processing",
                "message": "Stats updates are already being processed"
            }
        
        with cls._lock:
            if not cls._pending_updates:
                return {
                    "status": "no_updates",
                    "message": "No pending updates to process"
                }
            
            updates_to_process = list(cls._pending_updates)
            cls._pending_updates.clear()
            cls._is_processing = True
        
        try:
            logger.info(f"Force processing {len(updates_to_process)} pending stats updates")
            
            total_successful = 0
            total_failed = 0
            batch_size = 50
            
            for i in range(0, len(updates_to_process), batch_size):
                batch = updates_to_process[i:i + batch_size]
                successful, failed = cls._process_batch(batch)
                total_successful += successful
                total_failed += failed
            
            return {
                "status": "completed",
                "total_processed": len(updates_to_process),
                "successful": total_successful,
                "failed": total_failed
            }
            
        finally:
            cls._is_processing = False
    
    @classmethod
    @timed
    def get_status(cls) -> dict:
        """Get current status of the stats manager"""
        with cls._lock:
            pending_count = len(cls._pending_updates)
        
        return {
            "worker_active": cls._worker_thread and cls._worker_thread.is_alive() if cls._worker_thread else False,
            "is_processing": cls._is_processing,
            "pending_updates": pending_count,
            "worker_thread_name": cls._worker_thread.name if cls._worker_thread else None
        }
    
    @classmethod
    @timed
    def clear_pending_queue(cls) -> int:
        """
        Clear all pending stats updates (emergency use only)
        
        Returns:
            int: Number of cleared entries
        """
        with cls._lock:
            count = len(cls._pending_updates)
            cls._pending_updates.clear()
        
        logger.warning(f"Cleared {count} pending stats updates")
        return count

# Utility functions for manual management

@timed
def force_update_user_stats(user_id: int) -> bool:
    """
    Force immediate stats update for a user (use sparingly)
    
    Args:
        user_id: User ID to update
        
    Returns:
        bool: True if successful
    """
    try:
        db = SessionLocal()
        try:
            BackgroundTasks.refresh_user_stats(user_id, db)
            db.commit()
            return True
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Failed to force update stats for user {user_id}: {e}")
        return False

# Initialize the worker on module load
AsyncUserStatsManager.initialize()