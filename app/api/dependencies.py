from fastapi import Depends, HTTPException, status, Header, Security
from sqlalchemy.orm import Session
from app.database import get_db
from fastapi.security.api_key import APIKeyHeader
from app.config import settings
import logging

logger = logging.getLogger(__name__)

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

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Verify the API key from the request header.
    Raises HTTPException if the key is invalid or missing.
    """
    logger.debug(f"Received API key in header: {'Present' if api_key else 'Missing'}")
    
    if api_key is None:
        logger.warning("API key is missing from request")
        raise HTTPException(
            status_code=401,
            detail="API key is missing",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    if api_key != settings.API_KEY:
        logger.warning("Invalid API key provided")
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    logger.debug("API key verification successful")
    return api_key