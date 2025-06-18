from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db

async def get_current_user_id(user_id: int) -> int:
    """Validate user exists and return user_id"""
    # In a real application, this would validate from auth token
    # For now, we just return the user_id from request
    if user_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID"
        )
    return user_id