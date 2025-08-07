from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging
from app.database import SessionLocal
from app.core.scoring import SimilarityCalculator
from app.utils.constants import InteractionWeights
import numpy as np

logger = logging.getLogger(__name__)

class BackgroundTasks:
    """Background tasks for pre-computing recommendations and similarities"""
    
    @staticmethod
    def compute_all_user_similarities():
        """Compute similarities between all users"""
        db = SessionLocal()
        try:
            logger.info("Starting user similarity computation")
            
            # Get all active users
            users_query = text("""
                SELECT DISTINCT user_id FROM (
                    SELECT user_id FROM movie_ratings
                    UNION
                    SELECT user_id FROM movie_preferences
                ) active_users
                ORDER BY user_id
            """)
            
            user_ids = [row[0] for row in db.execute(users_query).fetchall()]
            logger.info(f"Found {len(user_ids)} active users")
            
            # Get all user interactions
            all_interactions = {}
            for user_id in user_ids:
                all_interactions[user_id] = BackgroundTasks._get_user_interactions(db, user_id)
            
            # Compute pairwise similarities
            similarity_calculator = SimilarityCalculator()
            computed_pairs = 0
            
            for i, user1_id in enumerate(user_ids):
                for user2_id in user_ids[i+1:]:
                    if user1_id < user2_id:
                        similarity, common_movies = similarity_calculator.calculate_user_similarity(
                            all_interactions[user1_id],
                            all_interactions[user2_id]
                        )
                        
                        # Store similarity if significant
                        if similarity > 0.1:  # Minimum threshold
                            BackgroundTasks._store_user_similarity(
                                db, user1_id, user2_id, similarity, common_movies
                            )
                            computed_pairs += 1
                
                if i % 100 == 0:
                    logger.info(f"Processed {i}/{len(user_ids)} users")
                    db.commit()
            
            db.commit()
            logger.info(f"Completed user similarity computation. Computed {computed_pairs} pairs.")
            
        except Exception as e:
            logger.error(f"Error computing user similarities: {e}")
            db.rollback()
        finally:
            db.close()
    
    @staticmethod
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
    def refresh_materialized_views():
        """Refresh all materialized views"""
        db = SessionLocal()
        try:
            logger.info("Refreshing materialized views")
            
            db.execute(text("REFRESH MATERIALIZED VIEW CONCURRENTLY user_interaction_summary"))
            db.commit()
            
            logger.info("Materialized views refreshed successfully")
            
        except Exception as e:
            logger.error(f"Error refreshing materialized views: {e}")
            db.rollback()
        finally:
            db.close()
    
    @staticmethod
    def refresh_user_stats(user_id: int, db: Optional[Session] = None):
        """
        Refresh stats for a specific user in real-time
        This is called after user interactions
        """
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True
            
        try:
            # Update the user's row in the materialized view
            refresh_query = text("""
                -- Delete existing row
                DELETE FROM user_interaction_summary WHERE user_id = :user_id;
                
                -- Insert fresh data
                INSERT INTO user_interaction_summary
                SELECT 
                    u.id as user_id,
                    COUNT(DISTINCT mp.movie_id) as total_preferences,
                    COUNT(DISTINCT mr.movie_id) as total_ratings,
                    AVG(mr.rating) as avg_rating,
                    COUNT(DISTINCT CASE WHEN mp.preference = 'LIKE' THEN mp.movie_id END) as liked_movies,
                    COUNT(DISTINCT CASE WHEN mp.preference = 'SUPERLIKE' THEN mp.movie_id END) as superliked_movies,
                    COUNT(DISTINCT CASE WHEN mp.preference = 'DISLIKE' THEN mp.movie_id END) as disliked_movies,
                    COUNT(DISTINCT CASE WHEN mr.rating >= 4 THEN mr.movie_id END) as high_rated_movies,
                    COUNT(DISTINCT CASE WHEN mr.rating <= 2 THEN mr.movie_id END) as low_rated_movies,
                    ARRAY_AGG(DISTINCT ug.genre_id) FILTER (WHERE ug.genre_id IS NOT NULL) as preferred_genres,
                    COUNT(DISTINCT m.matched_user_id) as total_matches,
                    CURRENT_TIMESTAMP as last_updated
                FROM users u
                LEFT JOIN movie_preferences mp ON u.id = mp.user_id
                LEFT JOIN movie_ratings mr ON u.id = mr.user_id
                LEFT JOIN user_genres ug ON u.id = ug.user_id
                LEFT JOIN (
                    SELECT user_id, matched_user_id FROM matches
                    UNION
                    SELECT matched_user_id as user_id, user_id as matched_user_id FROM matches
                ) m ON u.id = m.user_id
                WHERE u.id = :user_id
                GROUP BY u.id;
            """)
            
            db.execute(refresh_query, {'user_id': user_id})
            
            # Also invalidate recommendation cache for this user
            cache_clear_query = text("""
                DELETE FROM recommendation_cache 
                WHERE user_id = :user_id
            """)
            db.execute(cache_clear_query, {'user_id': user_id})
            
            db.commit()
            logger.info(f"Refreshed stats for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error refreshing user stats for user {user_id}: {e}")
            db.rollback()
        finally:
            if close_db:
                db.close()
    
    @staticmethod
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
    
    @staticmethod
    def _get_user_interactions(db: Session, user_id: int) -> Dict[int, float]:
        """Get user interactions as weights"""
        interactions = {}
        
        # Get ratings
        rating_query = text("""
            SELECT movie_id, rating FROM movie_ratings WHERE user_id = :user_id
        """)
        for row in db.execute(rating_query, {'user_id': user_id}).fetchall():
            interactions[row[0]] = InteractionWeights.get_weight(None, row[1])
        
        # Get preferences
        pref_query = text("""
            SELECT movie_id, preference FROM movie_preferences WHERE user_id = :user_id
        """)
        for row in db.execute(pref_query, {'user_id': user_id}).fetchall():
            weight = InteractionWeights.get_weight(row[1])
            if row[0] in interactions:
                interactions[row[0]] = max(interactions[row[0]], weight)
            else:
                interactions[row[0]] = weight
        
        return interactions
    
    @staticmethod
    def _get_movie_features(db: Session, movie_id: int) -> Dict:
        """Get movie features for similarity calculation"""
        # Basic movie info - updated to include vote_count
        movie_query = text("""
            SELECT 
                genre, popularity, release_date, 
                original_language, vote_average, vote_count
            FROM movies WHERE id = :movie_id
        """)
        
        movie_data = db.execute(movie_query, {'movie_id': movie_id}).fetchone()
        
        if not movie_data:
            return {}
        
        features = {
            'genres': [],
            'cast_ids': [],
            'production_companies': [],
            'language': movie_data[3],
            'popularity': float(movie_data[1]) if movie_data[1] else 0,
            'year': movie_data[2].year if movie_data[2] else None,
            'vote_average': float(movie_data[4]) if movie_data[4] else 0,
            'vote_count': int(movie_data[5]) if movie_data[5] else 0
        }
        
        # Get genres
        genre_query = text("""
            SELECT genre_id FROM movie_list_genres WHERE movie_id = :movie_id
        """)
        features['genres'] = [row[0] for row in db.execute(genre_query, {'movie_id': movie_id}).fetchall()]
        
        # Get top cast
        cast_query = text("""
            SELECT mcr.cast_id 
            FROM movie_cast_relations mcr
            JOIN movie_cast mc ON mcr.cast_id = mc.id
            WHERE mcr.movie_id = :movie_id
            ORDER BY mc.popularity DESC
            LIMIT 5
        """)
        features['cast_ids'] = [row[0] for row in db.execute(cast_query, {'movie_id': movie_id}).fetchall()]
        
        # Get production companies
        prod_query = text("""
            SELECT production_company_id 
            FROM production_company_relations 
            WHERE movie_id = :movie_id
        """)
        features['production_companies'] = [row[0] for row in db.execute(prod_query, {'movie_id': movie_id}).fetchall()]
        
        return features
    
    @staticmethod
    def _store_user_similarity(
        db: Session, 
        user1_id: int, 
        user2_id: int, 
        similarity: float,
        common_movies: int
    ):
        """Store user similarity in database"""
        query = text("""
            INSERT INTO user_similarities (user_id_1, user_id_2, similarity_score, common_movies, last_calculated)
            VALUES (:user1, :user2, :similarity, :common_movies, :now)
            ON CONFLICT (user_id_1, user_id_2) 
            DO UPDATE SET 
                similarity_score = :similarity,
                common_movies = :common_movies,
                last_calculated = :now
        """)
        
        db.execute(query, {
            'user1': min(user1_id, user2_id),
            'user2': max(user1_id, user2_id),
            'similarity': similarity,
            'common_movies': common_movies,
            'now': datetime.utcnow()
        })
    
    @staticmethod
    def _store_movie_similarity(
        db: Session,
        movie1_id: int,
        movie2_id: int,
        similarity: float
    ):
        """Store movie similarity in database"""
        query = text("""
            INSERT INTO movie_similarities (movie_id_1, movie_id_2, similarity_score, last_calculated)
            VALUES (:movie1, :movie2, :similarity, :now)
            ON CONFLICT (movie_id_1, movie_id_2)
            DO UPDATE SET 
                similarity_score = :similarity,
                last_calculated = :now
        """)
        
        db.execute(query, {
            'movie1': min(movie1_id, movie2_id),
            'movie2': max(movie1_id, movie2_id),
            'similarity': similarity,
            'now': datetime.utcnow()
        })

    @staticmethod
    def refresh_materialized_views():
        """Refresh all materialized views"""
        db = SessionLocal()
        try:
            logger.info("Refreshing materialized views")
            
            # Refresh user interaction summary
            db.execute(text("REFRESH MATERIALIZED VIEW CONCURRENTLY user_interaction_summary"))
            db.commit()
            
            logger.info("Materialized views refreshed successfully")
            
        except Exception as e:
            logger.error(f"Error refreshing materialized views: {e}")
            db.rollback()
        finally:
            db.close()
    
    @staticmethod
    def refresh_user_stats(user_id: int, db: Optional[Session] = None):
        """
        Refresh stats for a specific user in real-time
        This is called after user interactions
        """
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True
            
        try:
            # Update the user's row in the materialized view
            refresh_query = text("""
                -- Delete existing row
                DELETE FROM user_interaction_summary WHERE user_id = :user_id;
                
                -- Insert fresh data
                INSERT INTO user_interaction_summary
                SELECT 
                    u.id as user_id,
                    COUNT(DISTINCT mp.movie_id) as total_preferences,
                    COUNT(DISTINCT mr.movie_id) as total_ratings,
                    AVG(mr.rating) as avg_rating,
                    COUNT(DISTINCT CASE WHEN mp.preference = 'LIKE' THEN mp.movie_id END) as liked_movies,
                    COUNT(DISTINCT CASE WHEN mp.preference = 'SUPERLIKE' THEN mp.movie_id END) as superliked_movies,
                    COUNT(DISTINCT CASE WHEN mp.preference = 'DISLIKE' THEN mp.movie_id END) as disliked_movies,
                    COUNT(DISTINCT CASE WHEN mr.rating >= 4 THEN mr.movie_id END) as high_rated_movies,
                    COUNT(DISTINCT CASE WHEN mr.rating <= 2 THEN mr.movie_id END) as low_rated_movies,
                    ARRAY_AGG(DISTINCT ug.genre_id) FILTER (WHERE ug.genre_id IS NOT NULL) as preferred_genres,
                    COUNT(DISTINCT m.matched_user_id) as total_matches
                FROM users u
                LEFT JOIN movie_preferences mp ON u.id = mp.user_id
                LEFT JOIN movie_ratings mr ON u.id = mr.user_id
                LEFT JOIN user_genres ug ON u.id = ug.user_id
                LEFT JOIN (
                    SELECT user_id, matched_user_id FROM matches
                    UNION
                    SELECT matched_user_id as user_id, user_id as matched_user_id FROM matches
                ) m ON u.id = m.user_id
                WHERE u.id = :user_id
                GROUP BY u.id;
            """)
            
            db.execute(refresh_query, {'user_id': user_id})
            
            # Also invalidate recommendation cache for this user
            cache_clear_query = text("""
                DELETE FROM recommendation_cache 
                WHERE user_id = :user_id
            """)
            db.execute(cache_clear_query, {'user_id': user_id})
            
            db.commit()
            logger.info(f"Refreshed stats for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error refreshing user stats for user {user_id}: {e}")
            db.rollback()
        finally:
            if close_db:
                db.close()