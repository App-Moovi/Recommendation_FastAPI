from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.config import settings
from app.api.endpoints import recommendations, health
from app.background.scheduler import TaskScheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Initialize scheduler
task_scheduler = TaskScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Movie Recommender API")
    task_scheduler.start()
    yield
    # Shutdown
    logger.info("Shutting down Movie Recommender API")
    task_scheduler.shutdown()

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
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