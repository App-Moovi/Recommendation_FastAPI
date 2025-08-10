from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging
import time
import gc
from app.database import SessionLocal
from app.core.scoring import SimilarityCalculator
from app.utils.constants import InteractionWeights
from app.utils.job_protection import prevent_overlap
import numpy as np
from app.utils.logger import timed

logger = logging.getLogger(__name__)

class BackgroundTasks:
    """Background tasks for pre-computing recommendations and similarities"""
    
    @staticmethod
    @timed
    def compute_all_user_similarities():
        """
        DEPRECATED: This method has O(n²) complexity and will crash with large user bases.
        Use compute_user_similarities_batch() instead.
        """
        logger.error("compute_all_user_similarities() is deprecated due to O(n²) performance issues")
        raise NotImplementedError("Use compute_user_similarities_batch() instead")
    
    @staticmethod
    @timed
    @prevent_overlap("user_similarities_batch", timeout=7200)  # 2 hours max
    def compute_user_similarities_batch(batch_size: int = 100):
        """Process users in small batches to prevent memory issues"""
        db = SessionLocal()
        try:
            logger.info("Starting batch user similarity computation")
            
            # Get total user count
            count_query = text("""
                SELECT COUNT(DISTINCT user_id) FROM (
                    SELECT user_id FROM movie_ratings
                    UNION
                    SELECT user_id FROM movie_preferences
                ) u
            """)
            total_users = db.execute(count_query).scalar()
            
            if not total_users:
                logger.info("No active users found")
                return
            
            logger.info(f"Processing {total_users} users in batches of {batch_size}")
            
            # Process in batches
            processed_users = 0
            total_pairs = 0
            
            for offset in range(0, total_users, batch_size):
                batch_start_time = time.time()
                
                # Get batch of users
                batch_users_query = text("""
                    SELECT DISTINCT user_id FROM (
                        SELECT user_id FROM movie_ratings
                        UNION
                        SELECT user_id FROM movie_preferences
                    ) active_users
                    ORDER BY user_id
                    LIMIT :batch_size OFFSET :offset
                """)
                
                batch_users = [row[0] for row in db.execute(
                    batch_users_query, 
                    {'batch_size': batch_size, 'offset': offset}
                ).fetchall()]
                
                if not batch_users:
                    break
                
                # Process this batch
                pairs_created = BackgroundTasks._process_user_batch(db, batch_users)
                total_pairs += pairs_created
                processed_users += len(batch_users)
                
                # Commit after each batch
                db.commit()
                
                # Log progress
                batch_time = time.time() - batch_start_time
                logger.info(f"Processed batch {offset//batch_size + 1}: "
                           f"{processed_users}/{total_users} users, "
                           f"{pairs_created} new pairs, "
                           f"time: {batch_time:.2f}s")
                
                # Force garbage collection to free memory
                gc.collect()
                
                # Small delay to prevent overwhelming the database
                time.sleep(1)
            
            logger.info(f"Completed batch similarity computation. "
                       f"Processed {processed_users} users, "
                       f"created {total_pairs} similarity pairs")
            
        except Exception as e:
            logger.error(f"Error in batch similarity computation: {e}")
            db.rollback()
            raise
        finally:
            db.close()
    
    @staticmethod
    @timed
    def _process_user_batch(db: Session, user_ids: List[int]) -> int:
        """Process similarities for a small batch of users"""
        # Get interactions for this batch only
        interactions = {}
        
        for user_id in user_ids:
            user_interactions = BackgroundTasks._get_user_interactions(db, user_id)
            # Only keep users with significant interactions
            if len(user_interactions) >= 3:
                interactions[user_id] = user_interactions
        
        if len(interactions) < 2:
            return 0
        
        # Compute similarities within batch
        similarity_calculator = SimilarityCalculator()
        pairs_created = 0
        user_list = list(interactions.keys())
        
        for i, user1_id in enumerate(user_list):
            for j in range(i + 1, len(user_list)):
                user2_id = user_list[j]
                
                try:
                    similarity, common_movies = similarity_calculator.calculate_user_similarity(
                        interactions[user1_id],
                        interactions[user2_id]
                    )
                    
                    # Only store meaningful similarities
                    if similarity > 0.3 and common_movies >= 3:
                        BackgroundTasks._store_user_similarity(
                            db, user1_id, user2_id, similarity, common_movies
                        )
                        pairs_created += 1
                        
                except Exception as e:
                    logger.error(f"Error computing similarity between users {user1_id} and {user2_id}: {e}")
                    continue
        
        return pairs_created

    @staticmethod
    @timed
    def compute_movie_similarities():
        """Compute similarities between movies"""
        db = SessionLocal()
        try:
            logger.info("Starting movie similarity computation")
            
            # Get all movies with features
            movies_query = text("""
                SELECT DISTINCT m.id
                FROM movies m
                JOIN movie_list_genres mlg ON m.id = mlg.movie_id
                WHERE m.popularity > 0
                ORDER BY m.popularity DESC
                LIMIT 5000  -- Process top 5000 movies
            """)
            
            movie_ids = [row[0] for row in db.execute(movies_query).fetchall()]
            logger.info(f"Processing {len(movie_ids)} movies")
            
            # Get features for all movies
            movie_features = {}
            for movie_id in movie_ids:
                movie_features[movie_id] = BackgroundTasks._get_movie_features(db, movie_id)
            
            # Compute pairwise similarities
            similarity_calculator = SimilarityCalculator()
            computed_pairs = 0
            
            for i, movie1_id in enumerate(movie_ids):
                # Only compare with movies that share at least one genre
                related_movies_query = text("""
                    SELECT DISTINCT mlg2.movie_id
                    FROM movie_list_genres mlg1
                    JOIN movie_list_genres mlg2 ON mlg1.genre_id = mlg2.genre_id
                    WHERE mlg1.movie_id = :movie_id 
                        AND mlg2.movie_id > :movie_id
                        AND mlg2.movie_id = ANY(:movie_ids)
                """)
                
                related_movies = [
                    row[0] for row in db.execute(
                        related_movies_query,
                        {'movie_id': movie1_id, 'movie_ids': movie_ids}
                    ).fetchall()
                ]
                
                for movie2_id in related_movies:
                    similarity = similarity_calculator.calculate_movie_similarity(
                        movie_features[movie1_id],
                        movie_features[movie2_id]
                    )
                    
                    if similarity > 0.3:  # Minimum threshold
                        BackgroundTasks._store_movie_similarity(
                            db, movie1_id, movie2_id, similarity
                        )
                        computed_pairs += 1
                
                if i % 100 == 0:
                    logger.info(f"Processed {i}/{len(movie_ids)} movies")
                    db.commit()
            
            db.commit()
            logger.info(f"Completed movie similarity computation. Computed {computed_pairs} pairs.")
            
        except Exception as e:
            logger.error(f"Error computing movie similarities: {e}")
            db.rollback()
        finally:
            db.close()
    
    @staticmethod
    @timed
    def update_user_summary(user_id: int):
        session = SessionLocal()
        try:
            # Fetch data using raw SQL
            result = session.execute(text("""
                WITH
                preferences AS (
                    SELECT
                        COUNT(*) AS total_preferences,
                        COUNT(*) FILTER (WHERE preference = 'LIKE') AS liked_movies,
                        COUNT(*) FILTER (WHERE preference = 'SUPERLIKE') AS superliked_movies,
                        COUNT(*) FILTER (WHERE preference = 'DISLIKE') AS disliked_movies
                    FROM movie_preferences
                    WHERE user_id = :user_id
                ),
                ratings AS (
                    SELECT
                        COUNT(*) AS total_ratings,
                        AVG(rating) AS avg_rating,
                        COUNT(*) FILTER (WHERE rating >= 4) AS high_rated_movies,
                        COUNT(*) FILTER (WHERE rating < 3) AS low_rated_movies
                    FROM movie_ratings
                    WHERE user_id = :user_id
                ),
                genres AS (
                    SELECT ARRAY_AGG(genre_id) AS preferred_genres
                    FROM user_genres
                    WHERE user_id = :user_id
                ),
                matches AS (
                    SELECT COUNT(*) AS total_matches
                    FROM matches
                    WHERE user_id = :user_id
                )
                SELECT
                    p.total_preferences,
                    p.liked_movies,
                    p.superliked_movies,
                    p.disliked_movies,
                    r.total_ratings,
                    r.avg_rating,
                    r.high_rated_movies,
                    r.low_rated_movies,
                    g.preferred_genres,
                    m.total_matches
                FROM preferences p, ratings r, genres g, matches m;
            """), {"user_id": user_id}).mappings().first()

            # Upsert logic
            upsert_stmt = text("""
                INSERT INTO user_interaction_summary (
                    user_id,
                    total_preferences,
                    liked_movies,
                    superliked_movies,
                    disliked_movies,
                    total_ratings,
                    avg_rating,
                    high_rated_movies,
                    low_rated_movies,
                    preferred_genres,
                    total_matches,
                    last_updated
                ) VALUES (
                    :user_id,
                    :total_preferences,
                    :liked_movies,
                    :superliked_movies,
                    :disliked_movies,
                    :total_ratings,
                    :avg_rating,
                    :high_rated_movies,
                    :low_rated_movies,
                    :preferred_genres,
                    :total_matches,
                    :last_updated
                )
                ON CONFLICT (user_id)
                DO UPDATE SET
                    total_preferences = EXCLUDED.total_preferences,
                    liked_movies = EXCLUDED.liked_movies,
                    superliked_movies = EXCLUDED.superliked_movies,
                    disliked_movies = EXCLUDED.disliked_movies,
                    total_ratings = EXCLUDED.total_ratings,
                    avg_rating = EXCLUDED.avg_rating,
                    high_rated_movies = EXCLUDED.high_rated_movies,
                    low_rated_movies = EXCLUDED.low_rated_movies,
                    preferred_genres = EXCLUDED.preferred_genres,
                    total_matches = EXCLUDED.total_matches,
                    last_updated = EXCLUDED.last_updated;
            """)

            session.execute(upsert_stmt, {
                "user_id": user_id,
                **result,
                "last_updated": datetime.utcnow()
            })

            session.commit()
            logger.info(f"Updated summary for user_id={user_id}")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update summary for user_id={user_id}: {e}")
            raise
        finally:
            session.close()
    
    @staticmethod
    @timed
    def cleanup_expired_cache():
        """Clean up expired cache entries"""
        db = SessionLocal()
        try:
            logger.info("Cleaning up expired cache")
            
            # Delete expired recommendations
            db.execute(text("""
                DELETE FROM recommendation_cache 
                WHERE expires_at < :now
            """), {'now': datetime.utcnow()})
            
            # Delete expired potential matches
            db.execute(text("""
                DELETE FROM potential_matches 
                WHERE expires_at < :now
            """), {'now': datetime.utcnow()})
            
            db.commit()
            logger.info("Expired cache cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
            db.rollback()
        finally:
            db.close()