from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Optional
from datetime import datetime

from app.database import get_db
from app.models.schemas import (
    RecommendationResponse, 
    RecommendationRequest,
    MovieRecommendation
)
from app.background.user_stats import AsyncUserStatsManager
from app.core.recommendation_engine import RecommendationEngine
from app.api.dependencies import get_current_user_id
from app.config import settings

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

@router.post("/", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    db: Session = Depends(get_db)
):
    """
    Get movie recommendations for a user
    
    - **user_id**: The user requesting recommendations
    - **force_refresh**: Force regeneration of recommendations (default: False)
    - **include_movie_details**: Include movie details in response (default: True)
    """
    try:
        # Validate user exists
        user_check = db.execute(
            text("SELECT id FROM users WHERE id = :user_id"),
            {"user_id": request.user_id}
        ).fetchone()
        
        if not user_check:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Initialize recommendation engine
        engine = RecommendationEngine(db)

        # Generate recommendations
        recommendations = engine.generate_recommendations(
            user_id=request.user_id,
            count=settings.CACHE_SIZE,
            force_refresh=request.force_refresh
        )
        
        # Get the first batch
        batch_recommendations = recommendations[:settings.RECOMMENDATIONS_PER_REQUEST] if recommendations else []
        
        # Add movie details if requested
        if request.include_movie_details:
            batch_recommendations = _add_movie_details(db, batch_recommendations)
        
        # Get cache info
        cache_info = _get_cache_info(db, request.user_id)
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=batch_recommendations,
            batch_number=1,
            total_cached=cache_info['total'],
            expires_at=cache_info['expires_at']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{user_id}/next", response_model=RecommendationResponse)
async def get_next_recommendations(
    user_id: int,
    batch_number: int = Query(default=2, ge=1, le=2),
    include_movie_details: bool = Query(default=True),
    db: Session = Depends(get_db)
):
    """
    Get the next batch of cached recommendations
    
    - **user_id**: The user ID
    - **batch_number**: Which batch to retrieve (1 or 2)
    - **include_movie_details**: Include movie details in response
    """
    try:
        # Get cached recommendations for the specified batch
        cache_query = text("""
            SELECT 
                rc.movie_id, rc.score, rc.recommendation_reason,
                rc.matched_user_ids, rc.position
            FROM recommendation_cache rc
            WHERE rc.user_id = :user_id 
                AND rc.batch_number = :batch_number
                AND rc.expires_at > :now
            ORDER BY rc.position
        """)
        
        results = db.execute(
            cache_query,
            {
                'user_id': user_id, 
                'batch_number': batch_number,
                'now': datetime.utcnow()
            }
        ).fetchall()
        
        if not results:
            raise HTTPException(
                status_code=404, 
                detail="No cached recommendations found. Please request new recommendations."
            )
        
        recommendations = []
        for row in results:
            # Get potential matches
            pot_matches_query = text("""
                SELECT potential_match_user_id, match_condition, confidence_score
                FROM potential_matches
                WHERE user_id = :user_id AND movie_id = :movie_id
                    AND expires_at > :now
            """)
            
            pot_results = db.execute(
                pot_matches_query,
                {'user_id': user_id, 'movie_id': row[0], 'now': datetime.utcnow()}
            ).fetchall()
            
            rec = MovieRecommendation(
                movie_id=row[0],
                score=float(row[1]),
                recommendation_reason=row[2],
                matched_user_ids=row[3] or [],
                potential_matches=[
                    {
                        'user_id': pm[0],
                        'match_condition': pm[1],
                        'confidence_score': float(pm[2])
                    } for pm in pot_results
                ]
            )
            recommendations.append(rec)
        
        # Add movie details if requested
        if include_movie_details:
            recommendations = _add_movie_details(db, recommendations)
        
        # Get cache info
        cache_info = _get_cache_info(db, user_id)
        
        return RecommendationResponse(
            user_id=user_id,
            recommendations=recommendations,
            batch_number=batch_number,
            total_cached=cache_info['total'],
            expires_at=cache_info['expires_at']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{user_id}/queue-stats-update")
async def queue_user_stats_update(
    user_id: int,
    db: Session = Depends(get_db)
):
    """
    Queue a user's statistics for asynchronous update
    
    - **user_id**: The user ID whose stats need updating
    """
    try:
        # Validate user exists
        user_check = db.execute(
            text("SELECT id FROM users WHERE id = :user_id"),
            {"user_id": user_id}
        ).fetchone()
        
        if not user_check:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Queue the stats update
        success = AsyncUserStatsManager.queue_user_stats_update(user_id)
        
        if not success:
            raise HTTPException(
                status_code=503, 
                detail="Failed to queue stats update. Stats service may be unavailable."
            )
        
        # Get current pending count
        pending_count = AsyncUserStatsManager.get_pending_updates_count()
        
        return {
            "status": "success",
            "message": f"Stats update queued for user {user_id}",
            "pending_updates": pending_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
def _add_movie_details(db: Session, recommendations: List[MovieRecommendation]) -> List[MovieRecommendation]:
    """Add movie details to recommendations"""
    movie_ids = [rec.movie_id for rec in recommendations]
    
    if not movie_ids:
        return recommendations
    
    # Fetch movie details
    movie_query = text("""
        SELECT 
            id, title, overview, genre, release_date,
            vote_average, poster_path, backdrop_path, 
            original_title, original_language, adult, 
            popularity, video, vote_count
        FROM movies
        WHERE id = ANY(:movie_ids)
    """)
    
    results = db.execute(movie_query, {'movie_ids': movie_ids}).fetchall()
    
    # Create lookup dictionary
    movie_details = {
        row[0]: {
            'title': row[1],
            'overview': row[2],
            'genre': row[3],
            'release_date': row[4].isoformat() if row[4] else None,
            'vote_average': float(row[5]) if row[5] else None,
            'poster_path': row[6],
            'backdrop_path': row[7],
            'original_title': row[8],
            'original_language': row[9],
            'adult': row[10],
            'popularity': float(row[11]) if row[11] else None,
            'video': row[12],
            'vote_count': row[13]
        }
        for row in results
    }
    
    # Update recommendations with details
    for rec in recommendations:
        if rec.movie_id in movie_details:
            details = movie_details[rec.movie_id]
            rec.score = float("%0.4f" % rec.score)
            rec.title = details['title']
            rec.overview = details['overview']
            rec.genre = details['genre']
            rec.release_date = details['release_date']
            rec.vote_average = details['vote_average']
            rec.poster_path = details['poster_path']
            rec.backdrop_path = details['backdrop_path']
            rec.original_title = details['original_title']
            rec.original_language = details['original_language']
            rec.adult = details['adult']
            rec.popularity = details['popularity']
            rec.video = details['video']
            rec.vote_count = details['vote_count']
    
    return sorted(recommendations, key=lambda rec: rec.score, reverse=True)

def _get_cache_info(db: Session, user_id: int) -> dict:
    """Get cache information for a user"""
    cache_query = text("""
        SELECT COUNT(*), MAX(expires_at)
        FROM recommendation_cache
        WHERE user_id = :user_id AND expires_at > :now
    """)
    
    result = db.execute(
        cache_query,
        {'user_id': user_id, 'now': datetime.utcnow()}
    ).fetchone()
    
    return {
        'total': result[0] if result else 0,
        'expires_at': result[1] if result and result[1] else datetime.utcnow()
    }

async def _check_and_create_match(
    db: Session,
    user_id: int,
    movie_id: int,
    interaction_type: str,
    rating: Optional[float]
) -> Optional[dict]:
    """Check if interaction triggers a match and create it"""
    # Check potential matches for this movie
    pot_match_query = text("""
        SELECT potential_match_user_id, match_condition, confidence_score
        FROM potential_matches
        WHERE user_id = :user_id AND movie_id = :movie_id
            AND expires_at > :now
    """)
    
    pot_matches = db.execute(
        pot_match_query,
        {'user_id': user_id, 'movie_id': movie_id, 'now': datetime.utcnow()}
    ).fetchall()
    
    for match in pot_matches:
        matched_user_id, condition, confidence = match
        
        # Check if the interaction satisfies the match condition
        should_match = False
        
        if condition == 'superlike' and interaction_type == 'SUPERLIKE':
            should_match = True
        elif condition == 'like' and interaction_type == 'LIKE':
            should_match = True
        elif condition == 'dislike' and interaction_type == 'DISLIKE':
            should_match = True
        elif condition == 'rating_5' and rating == 5:
            should_match = True
        elif condition == 'rating_4+' and rating and rating >= 4:
            should_match = True
        elif condition == 'rating_3+' and rating and rating >= 3:
            should_match = True
        
        if should_match and confidence >= settings.MATCH_THRESHOLD:
            # Create match request
            match_request_query = text("""
                INSERT INTO match_requests (user_id, matched_user_id, status, created_at)
                VALUES (:user_id, :matched_user_id, 'PENDING', CURRENT_TIMESTAMP)
                ON CONFLICT DO NOTHING
                RETURNING id
            """)
            
            result = db.execute(
                match_request_query,
                {'user_id': user_id, 'matched_user_id': matched_user_id}
            ).fetchone()
            
            if result:
                db.commit()
                return {
                    'matched_user_id': matched_user_id,
                    'confidence_score': float(confidence),
                    'match_condition': condition
                }
    
    return None