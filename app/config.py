from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # API Key
    API_KEY: str = "moovii_engineogebvnotkjzylctjpbriujjiqykwcmtj"
    
    # Database
    DATABASE_URL: str = "postgresql://zaroor:Karthik%400306@moovii-database.postgres.database.azure.com:5432/moovii"
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Movie Recommender API"

    # Recommendation Settings
    RECOMMENDATIONS_PER_REQUEST: int = 10
    CACHE_SIZE: int = 20

    # Matching Settings
    MATCH_THRESHOLD: float = 0.5  # 80% similarity threshold

    # Background Jobs
    ENABLE_BACKGROUND_JOBS: bool = True
    
    class Config:
        env_file = ".env"


settings = Settings()
