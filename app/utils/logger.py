import logging

logger = logging.getLogger(__name__)

def timed(func):
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function '{func.__name__}' took {(end_time - start_time).__round__(6)} seconds or {((end_time - start_time) * 1000).__round__(4)} milliseconds to execute.")
        return result
    return wrapper