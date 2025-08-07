from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import sys

from app.config import settings
from app.api.endpoints import recommendations, health
from app.background.scheduler import TaskScheduler
from app.api.dependencies import verify_api_key
from app.background.user_stats import AsyncUserStatsManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

logger = logging.getLogger(__name__)

# Initialize scheduler
task_scheduler = TaskScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Startup
        logger.info("Starting Movie Recommender API")
        logger.info(f"API Key configured: {'Yes' if settings.API_KEY else 'No'}")
        task_scheduler.start()
        AsyncUserStatsManager.initialize()
        
        yield
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Movie Recommender API")
        task_scheduler.shutdown()
        AsyncUserStatsManager.shutdown()

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
    dependencies=[Depends(verify_api_key)]  # Apply API key verification to all endpoints
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    recommendations.router,
    prefix=settings.API_V1_STR
)
app.include_router(
    health.router,
    prefix=settings.API_V1_STR
)

@app.get("/")
async def root():
    return {
        "message": "Movie Recommender API",
        "version": "1.0.0",
        "docs": f"{settings.API_V1_STR}/docs"
    }