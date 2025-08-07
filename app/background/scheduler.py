from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from app.background.tasks import BackgroundTasks
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class TaskScheduler:
    """Scheduler for background tasks"""
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
    
    def start(self):
        """Start the scheduler with configured tasks"""
        if not settings.ENABLE_BACKGROUND_JOBS:
            logger.info("Background jobs are disabled")
            return
        
        # EMERGENCY FIX: Commented out dangerous jobs that cause O(n²) performance issues
        # DO NOT RE-ENABLE without implementing batch processing
        
        # # Daily tasks at midnight - DISABLED DUE TO PERFORMANCE ISSUES
        # self.scheduler.add_job(
        #     BackgroundTasks.compute_all_user_similarities,
        #     CronTrigger(minute='*/30'),  # This is extremely dangerous!
        #     id='compute_user_similarities',
        #     name='Compute User Similarities',
        #     replace_existing=True
        # )
        
        # Safe job: Compute user similarities in batches (daily at 3 AM)
        self.scheduler.add_job(
            BackgroundTasks.compute_user_similarities_batch,
            CronTrigger(hour=3, minute=0),  # Daily at 3 AM, not every 30 minutes!
            id='compute_user_similarities_batch',
            name='Compute User Similarities (Batch)',
            max_instances=1,  # Critical: prevent overlaps
            coalesce=True,
            replace_existing=True
        )
        
        # Safe job: Compute movie similarities (daily at 4 AM)
        self.scheduler.add_job(
            BackgroundTasks.compute_movie_similarities,
            CronTrigger(hour=4, minute=0),  # Daily at 4 AM
            id='compute_movie_similarities',
            name='Compute Movie Similarities',
            max_instances=1,  # Prevent overlaps
            coalesce=True,
            replace_existing=True
        )
        
        # Safe job: Cleanup expired cache (daily at 2 AM)
        self.scheduler.add_job(
            BackgroundTasks.cleanup_expired_cache,
            CronTrigger(hour=2, minute=0),  # Daily at 2 AM
            id='cleanup_cache',
            name='Cleanup Expired Cache',
            max_instances=1,  # Prevent overlaps
            coalesce=True,
            replace_existing=True
        )
        
        self.scheduler.start()
        logger.info("Background task scheduler started with safe job configuration")
    
    def shutdown(self):
        """Shutdown the scheduler"""
        self.scheduler.shutdown()
        logger.info("Background task scheduler stopped")