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
        
        # Daily tasks at midnight
        self.scheduler.add_job(
            BackgroundTasks.compute_all_user_similarities,
            CronTrigger(minute='*/30'),
            id='compute_user_similarities',
            name='Compute User Similarities',
            replace_existing=True
        )
        
        self.scheduler.add_job(
            BackgroundTasks.compute_movie_similarities,
            CronTrigger(hour=1, minute=0),
            id='compute_movie_similarities',
            name='Compute Movie Similarities',
            replace_existing=True
        )
        
        # Refresh materialized views every 6 hours
        self.scheduler.add_job(
            BackgroundTasks.refresh_materialized_views,
            CronTrigger(hour='*/6'),
            id='refresh_views',
            name='Refresh Materialized Views',
            replace_existing=True
        )
        
        # Cleanup expired cache every hour
        self.scheduler.add_job(
            BackgroundTasks.cleanup_expired_cache,
            CronTrigger(minute=0),
            id='cleanup_cache',
            name='Cleanup Expired Cache',
            replace_existing=True
        )
        
        self.scheduler.start()
        logger.info("Background task scheduler started")
    
    def shutdown(self):
        """Shutdown the scheduler"""
        self.scheduler.shutdown()
        logger.info("Background task scheduler stopped")