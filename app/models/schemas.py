from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class PotentialMatch(BaseModel):
    user_id: int
    match_condition: str
    confidence_score: float

class MovieRecommendation(BaseModel):
    movie_id: int
    score: float
    recommendation_reason: Optional[str] = None
    matched_user_ids: List[int] = Field(default_factory=list)
    potential_matches: List[PotentialMatch] = Field(default_factory=list)
    
    # Optional movie details
    title: Optional[str] = None
    original_title: Optional[str] = None
    overview: Optional[str] = None
    adult: Optional[bool] = None
    original_language: Optional[str] = None
    genre: Optional[str] = None
    popularity: Optional[float] = None
    video: Optional[bool] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None
    vote_count: Optional[int] = None
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[MovieRecommendation]
    batch_number: int
    total_cached: int
    expires_at: datetime

class UserMatchResponse(BaseModel):
    matched_user_id: int
    similarity_score: float
    common_movies: int
    match_reason: str

class RecommendationRequest(BaseModel):
    user_id: int
    force_refresh: bool = False
    include_movie_details: bool = True
