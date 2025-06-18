from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.database import get_db

router = APIRouter(prefix="/health", tags=["health"])

@router.get("/")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "service": "movie-recommender"}

@router.get("/db")
async def database_health(db: Session = Depends(get_db)):
    """Check database connectivity"""
    try:
        result = db.execute(text("SELECT 1")).fetchone()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "database": "disconnected", "error": str(e)}