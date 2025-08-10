from typing import List, Dict, Tuple, Set
from sqlalchemy.orm import Session
from sqlalchemy import text, bindparam
from datetime import datetime, timedelta
import logging
from app.core.scoring import RecommendationScorer, SimilarityCalculator
from app.core.diversity import DiversityOptimizer
from app.core.user_matching import UserMatcher
from app.models.schemas import MovieRecommendation, PotentialMatch
from app.utils.constants import InteractionWeights, MatchingThresholds
from app.config import settings
from app.utils.logger import timed

logger = logging.getLogger(__name__)

class RecommendationEngineError(Exception):
    """Custom exception for recommendation engine errors"""
    pass

class RecommendationEngine:
    """Main recommendation engine using hybrid approach"""
    
    def __init__(self, db: Session):
        self.db = db
        self.scorer = RecommendationScorer(db)
        self.diversity_optimizer = DiversityOptimizer()
        self.user_matcher = UserMatcher(db)
        self.similarity_calculator = SimilarityCalculator()
    
    @timed
    def generate_recommendations(
        self, 
        user_id: int, 
        count: int,
        force_refresh: bool = False,
        weights: Dict[str, float] = None
    ) -> List[MovieRecommendation]:
        """
        Generate movie recommendations for a user
        
        Args:
            user_id: The user to generate recommendations for
            count: Number of recommendations to generate (default: 40)
            force_refresh: Force regeneration of recommendations
            weights: Custom weights for scoring components
        """
        try:
            # Check if we need to use cached recommendations
            if not force_refresh:
                cached = self._get_cached_recommendations(user_id)
                if cached and len(cached) >= count:
                    logger.info(f"Returning {count} cached recommendations for user {user_id}")
                    return cached[:count]
            
            # Get user profile using materialized view
            user_profile = self._get_user_profile_optimized(user_id)
            print(user_profile)
            
            if not user_profile:
                raise RecommendationEngineError(f"User profile not found for user_id: {user_id}")
            
            # Get candidate movies
            candidate_movies = self._get_candidate_movies(user_id, user_profile)
            
            if not candidate_movies:
                raise RecommendationEngineError(f"No candidate movies found for user {user_id}")
            
            logger.info(f"Found {len(candidate_movies)} candidate movies for user {user_id}")
            
            # Score all candidates
            scored_movies = []
            for movie_id in candidate_movies:
                try:
                    movie_features = self._get_movie_features(movie_id)
                    if not movie_features:
                        continue
                    
                    # Use improved scoring with better weights
                    score, influencing_users, reason = self.scorer.score_movie_for_user(
                        user_id,
                        movie_id,
                        user_profile['interactions'],
                        user_profile['similar_users'],
                        movie_features,
                        movie_features.get('popularity', 0),
                        user_profile['genres'],
                        weights
                    )
                    
                    # Get potential matches for this movie
                    potential_matches = self.user_matcher.get_potential_matches(
                        user_id, movie_id, influencing_users
                    )
                    
                    scored_movies.append({
                        'movie_id': movie_id,
                        'score': score,
                        'reason': reason,
                        'influencing_users': influencing_users,
                        'potential_matches': potential_matches,
                        'features': movie_features
                    })
                except Exception as e:
                    logger.error(f"Error scoring movie {movie_id} for user {user_id}: {str(e)}")
                    continue
            
            if not scored_movies:
                raise RecommendationEngineError(f"Failed to score any movies for user {user_id}")
            
            # Sort by score
            scored_movies.sort(key=lambda x: x['score'], reverse=True)
            
            logger.info(f"Scored {len(scored_movies)} movies, top score: {scored_movies[0]['score']:.3f}")
            
            # Apply diversity optimization
            diverse_movies = self.diversity_optimizer.optimize_diversity(
                scored_movies[:count * 2],  # Consider 2x candidates
                count
            )
            
            # Convert to MovieRecommendation objects
            recommendations = []
            for idx, movie_data in enumerate(diverse_movies):
                rec = MovieRecommendation(
                    movie_id=movie_data['movie_id'],
                    score=movie_data['score'],
                    recommendation_reason=movie_data['reason'],
                    matched_user_ids=movie_data['influencing_users'],
                    potential_matches=[
                        PotentialMatch(
                            user_id=pm['user_id'],
                            match_condition=pm['condition'],
                            confidence_score=pm['confidence']
                        ) for pm in movie_data['potential_matches']
                    ]
                )
                recommendations.append(rec)
            
            # Cache recommendations
            self._cache_recommendations(user_id, recommendations)
            
            return recommendations
            
        except RecommendationEngineError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in generate_recommendations for user {user_id}: {str(e)}")
            raise RecommendationEngineError(f"Failed to generate recommendations: {str(e)}")
    
    @timed
    def _get_user_profile_optimized(self, user_id: int) -> Dict:
        """Get comprehensive user profile using materialized view for better performance"""
        try:
            profile = {
                'user_id': user_id,
                'interactions': {},
                'genres': [],
                'similar_users': [],
                'language_preferences': [],
                'stats': {}
            }
            
            # First, get stats from materialized view
            stats_query = text("""
                SELECT 
                    total_preferences,
                    total_ratings,
                    avg_rating,
                    liked_movies,
                    superliked_movies,
                    disliked_movies,
                    preferred_genres
                FROM user_interaction_summary
                WHERE user_id = :user_id
            """)
            
            stats_result = self.db.execute(stats_query, {'user_id': user_id}).fetchone()
            
            if stats_result:
                profile['stats'] = {
                    'total_preferences': stats_result[0] or 0,
                    'total_ratings': stats_result[1] or 0,
                    'avg_rating': float(stats_result[2] or 0),
                    'liked_movies': stats_result[3] or 0,
                    'superliked_movies': stats_result[4] or 0,
                    'disliked_movies': stats_result[5] or 0
                }
                profile['genres'] = stats_result[6] or []
            
            # Get detailed user interactions
            interactions_query = text("""
                SELECT movie_id, rating, NULL as preference FROM movie_ratings WHERE user_id = :user_id
                UNION ALL
                SELECT movie_id, NULL as rating, preference FROM movie_preferences WHERE user_id = :user_id
            """)
            
            results = self.db.execute(interactions_query, {'user_id': user_id}).fetchall()
            
            for row in results:
                movie_id = row[0]
                if row[1] is not None:  # Rating
                    weight = InteractionWeights.get_weight(None, row[1])
                else:  # Preference
                    weight = InteractionWeights.get_weight(row[2])
                
                # Use the highest weight if movie has both rating and preference
                if movie_id in profile['interactions']:
                    profile['interactions'][movie_id] = max(profile['interactions'][movie_id], weight)
                else:
                    profile['interactions'][movie_id] = weight
            
            # Get similar users using the SQL function
            similar_users_query = text("""
                WITH matched_users AS (
                    SELECT 
                        CASE 
                            WHEN user_id = :user_id THEN matched_user_id 
                            ELSE user_id 
                        END as matched_user_id
                    FROM matches
                    WHERE :user_id IN (user_id, matched_user_id)
                )
                SELECT 
                    mu.matched_user_id,
                    get_user_similarity(:user_id, mu.matched_user_id) as similarity_score
                FROM matched_users mu
                WHERE get_user_similarity(:user_id, mu.matched_user_id) >= :threshold
                ORDER BY similarity_score DESC
                LIMIT 20
            """)
            
            similar_results = self.db.execute(
                similar_users_query, 
                {'user_id': user_id, 'threshold': settings.MATCH_THRESHOLD}
            ).fetchall()
            
            profile['similar_users'] = [(row[0], float(row[1])) for row in similar_results if row[1] is not None]
            
            logger.info(f"User {user_id} profile: {len(profile['interactions'])} interactions, "
                       f"{len(profile['similar_users'])} matched similar users, "
                       f"{len(profile['genres'])} preferred genres")
            
            # Get language preferences
            lang_query = text("""
                SELECT language_code FROM user_languages WHERE user_id = :user_id
            """)
            profile['language_preferences'] = [row[0] for row in self.db.execute(lang_query, {'user_id': user_id}).fetchall()]
            
            logger.info(f"User {user_id} Profile: {profile}")
            return profile
            
        except Exception as e:
            logger.error(f"Error getting user profile for user {user_id}: {str(e)}")
            raise RecommendationEngineError(f"Failed to get user profile: {str(e)}")
    
    @timed
    def _get_candidate_movies(self, user_id: int, user_profile: Dict) -> Set[int]:
        """Get candidate movies for recommendation with improved genre focus"""
        try:
            candidates = set()
            
            # Exclude already interacted movies
            interacted_movies = set(user_profile['interactions'].keys())
            
            # Check if this is a cold start scenario
            is_cold_start = len(user_profile['interactions']) < 10 or not user_profile['similar_users']
            
            # 1. Movies from preferred genres (HIGH PRIORITY)
            if user_profile['genres']:
                # For better genre-based recommendations, get more candidates
                limit = 500 if is_cold_start else 300
                
                genre_movies_query = text("""
                    SELECT mlg.movie_id, COUNT(DISTINCT mlg.genre_id) as matching_genres, m.popularity
                    FROM movie_genre_links mlg
                    JOIN movies m ON mlg.movie_id = m.id
                    WHERE mlg.genre_id = ANY(:genre_ids)
                        AND mlg.movie_id NOT IN :interacted_movies
                        AND m.adult != TRUE
                        AND m.original_language = 'en'
                        AND m.vote_average >= 1.0  -- Quality threshold
                    GROUP BY mlg.movie_id, m.popularity
                    ORDER BY matching_genres DESC, m.popularity DESC
                    LIMIT :limit
                """).bindparams(bindparam("interacted_movies", expanding=True))
                
                results = self.db.execute(
                    genre_movies_query,
                    {
                        'genre_ids': user_profile['genres'],
                        'interacted_movies': list(interacted_movies) or [-1],
                        'limit': limit
                    }
                ).fetchall()
                
                candidates.update(row[0] for row in results)
                logger.info(f"Added {len(results)} movies from preferred genres")
            
            # 2. Movies from similar movies (using movie_similarities table)
            if user_profile['interactions']:
                # Get movies similar to user's highly rated movies
                similar_movies_query = text("""
                    SELECT DISTINCT 
                        CASE 
                            WHEN ms.movie_id_1 = ANY(:liked_movies) THEN ms.movie_id_2
                            ELSE ms.movie_id_1
                        END as similar_movie_id,
                        ms.similarity_score
                    FROM movie_similarities ms
                    WHERE (ms.movie_id_1 = ANY(:liked_movies) OR ms.movie_id_2 = ANY(:liked_movies))
                        AND ms.similarity_score >= 0.5
                        AND CASE 
                            WHEN ms.movie_id_1 = ANY(:liked_movies) THEN ms.movie_id_2
                            ELSE ms.movie_id_1
                        END NOT IN :interacted_movies
                    ORDER BY ms.similarity_score DESC
                    LIMIT 200
                """).bindparams(bindparam("interacted_movies", expanding=True))
                
                # Get user's liked movies (positive interactions)
                liked_movies = [movie_id for movie_id, score in user_profile['interactions'].items() if score > 1.0]
                
                if liked_movies:
                    results = self.db.execute(
                        similar_movies_query,
                        {
                            'liked_movies': liked_movies[:50],  # Top 50 liked movies
                            'interacted_movies': list(interacted_movies) or [-1]
                        }
                    ).fetchall()
                    
                    candidates.update(row[0] for row in results)
                    logger.info(f"Added {len(results)} movies from movie similarities")
            
            # 3. Movies from similar users (if not cold start)
            if user_profile['similar_users'] and not is_cold_start:
                similar_users_movies_query = text("""
                    SELECT DISTINCT m.movie_id, COUNT(DISTINCT m.user_id) as user_count
                    FROM (
                        SELECT movie_id, user_id FROM movie_ratings 
                        WHERE user_id = ANY(:user_ids) AND rating >= 4
                        UNION
                        SELECT movie_id, user_id FROM movie_preferences
                        WHERE user_id = ANY(:user_ids) 
                            AND preference IN ('LIKE', 'SUPERLIKE')
                    ) m
                    WHERE m.movie_id NOT IN :interacted_movies
                    GROUP BY m.movie_id
                    ORDER BY user_count DESC
                    LIMIT 200
                """).bindparams(bindparam("interacted_movies", expanding=True))
                
                similar_user_ids = [u[0] for u in user_profile['similar_users']]
                results = self.db.execute(
                    similar_users_movies_query,
                    {
                        'user_ids': similar_user_ids,
                        'interacted_movies': list(interacted_movies) or [-1]
                    }
                ).fetchall()
                
                candidates.update(row[0] for row in results)
                logger.info(f"Added {len(results)} movies from similar users")
            
            # 4. Popular movies in user's genres with language preference
            if user_profile['genres']:
                popular_genre_query = text("""
                    SELECT m.id
                    FROM movies m
                    JOIN movie_genre_links mlg ON m.id = mlg.movie_id
                    WHERE mlg.genre_id = ANY(:genre_ids)
                        AND m.id NOT IN :interacted_movies
                        AND m.vote_average >= 6.0
                        AND m.vote_count >= 100
                        AND m.adult != TRUE
                        AND m.original_language = 'en'
                        AND (:has_lang_pref = FALSE OR m.original_language = ANY(:lang_prefs))
                    GROUP BY m.id
                    ORDER BY m.popularity DESC
                    LIMIT 150
                """).bindparams(bindparam("interacted_movies", expanding=True))
                
                results = self.db.execute(
                    popular_genre_query,
                    {
                        'genre_ids': user_profile['genres'],
                        'interacted_movies': list(interacted_movies) or [-1],
                        'has_lang_pref': len(user_profile['language_preferences']) > 0,
                        'lang_prefs': user_profile['language_preferences'] or ['en']
                    }
                ).fetchall()
                
                candidates.update(row[0] for row in results)
                logger.info(f"Added {len(results)} popular movies from user's genres")
            
            # 5. Trending movies (smaller portion)
            trending_limit = 30
            trending_query = text("""
                SELECT id
                FROM movies 
                WHERE id NOT IN :interacted_movies
                    -- AND vote_average >= 6.0 -- Filter out low-rated movies
                    AND original_language = 'en'
                    AND adult != TRUE
                ORDER BY popularity DESC
                LIMIT :limit
            """).bindparams(bindparam("interacted_movies", expanding=True))
            
            results = self.db.execute(
                trending_query,
                {'interacted_movies': list(interacted_movies) or [-1], 'limit': trending_limit}
            ).fetchall()
            
            candidates.update(row[0] for row in results)
            logger.info(f"Added {len(results)} trending movies")
            
            logger.info(f"Total candidates: {len(candidates)} (Cold start: {is_cold_start})")
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error getting candidate movies for user {user_id}: {str(e)}")
            raise RecommendationEngineError(f"Failed to get candidate movies: {str(e)}")
    
    @timed
    def _get_movie_features(self, movie_id: int) -> Dict:
        """Get comprehensive features for a movie"""
        try:
            # Get basic movie info
            movie_query = text("""
                SELECT 
                    m.id, m.title, m.overview, m.genre, m.popularity,
                    m.vote_average, m.release_date, m.original_language,
                    m.runtime, m.poster_path, m.backdrop_path, m.vote_count
                FROM movies m
                WHERE m.id = :movie_id
            """)
            
            movie_result = self.db.execute(movie_query, {'movie_id': movie_id}).fetchone()
            
            if not movie_result:
                return {}
            
            features = {
                'movie_id': movie_result[0],
                'title': movie_result[1],
                'overview': movie_result[2],
                'genre': movie_result[3],
                'popularity': movie_result[4] or 0,
                'vote_average': movie_result[5] or 0,
                'release_date': movie_result[6],
                'language': movie_result[7],
                'runtime': movie_result[8],
                'poster_path': movie_result[9],
                'backdrop_path': movie_result[10],
                'vote_count': movie_result[11] or 0,
                'year': movie_result[6].year if movie_result[6] else None
            }
            
            # Get genres
            genre_query = text("""
                SELECT genre_id FROM movie_list_genres WHERE movie_id = :movie_id
            """)
            features['genres'] = [row[0] for row in self.db.execute(genre_query, {'movie_id': movie_id}).fetchall()]
            
            # Get cast IDs (top 5)
            cast_query = text("""
                SELECT mcr.cast_id 
                FROM movie_cast_relations mcr
                JOIN movie_cast mc ON mcr.cast_id = mc.id
                WHERE mcr.movie_id = :movie_id
                ORDER BY mc.popularity DESC
                LIMIT 5
            """)
            features['cast_ids'] = [row[0] for row in self.db.execute(cast_query, {'movie_id': movie_id}).fetchall()]
            
            # Get production companies
            prod_query = text("""
                SELECT production_company_id 
                FROM production_company_relations 
                WHERE movie_id = :movie_id
            """)
            features['production_companies'] = [row[0] for row in self.db.execute(prod_query, {'movie_id': movie_id}).fetchall()]
            
            return features
            
        except Exception as e:
            logger.error(f"Error getting movie features for movie {movie_id}: {str(e)}")
            return {}
    
    @timed
    def _get_cached_recommendations(self, user_id: int) -> List[MovieRecommendation]:
        """Get cached recommendations if available"""
        try:
            cache_query = text("""
                SELECT 
                    rc.movie_id, rc.score, rc.recommendation_reason,
                    rc.matched_user_ids, rc.position,
                    m.title, m.overview, m.genre, m.release_date,
                    m.vote_average, m.poster_path
                FROM recommendation_cache rc
                JOIN movies m ON rc.movie_id = m.id
                WHERE rc.user_id = :user_id 
                    AND rc.expires_at > :now
                ORDER BY rc.batch_number, rc.position
            """)
            
            results = self.db.execute(
                cache_query,
                {'user_id': user_id, 'now': datetime.utcnow()}
            ).fetchall()
            
            recommendations = []
            for row in results:
                # Get potential matches for this cached recommendation
                pot_matches_query = text("""
                    SELECT potential_match_user_id, match_condition, confidence_score
                    FROM potential_matches
                    WHERE user_id = :user_id AND movie_id = :movie_id
                        AND expires_at > :now
                """)
                
                pot_results = self.db.execute(
                    pot_matches_query,
                    {'user_id': user_id, 'movie_id': row[0], 'now': datetime.utcnow()}
                ).fetchall()
                
                potential_matches = [
                    PotentialMatch(
                        user_id=pm[0],
                        match_condition=pm[1],
                        confidence_score=pm[2]
                    ) for pm in pot_results
                ]
                
                rec = MovieRecommendation(
                    movie_id=row[0],
                    score=float(row[1]),
                    recommendation_reason=row[2],
                    matched_user_ids=row[3] or [],
                    potential_matches=potential_matches,
                    title=row[5],
                    overview=row[6],
                    genre=row[7],
                    release_date=row[8].isoformat() if row[8] else None,
                    vote_average=float(row[9]) if row[9] else None,
                    poster_path=row[10]
                )
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting cached recommendations for user {user_id}: {str(e)}")
            return []
    
    @timed
    def _cache_recommendations(self, user_id: int, recommendations: List[MovieRecommendation]):
        """Cache recommendations for future use"""
        try:
            # Clear old cache
            clear_query = text("""
                DELETE FROM recommendation_cache WHERE user_id = :user_id
            """)
            self.db.execute(clear_query, {'user_id': user_id})
            
            clear_matches_query = text("""
                DELETE FROM potential_matches WHERE user_id = :user_id
            """)
            self.db.execute(clear_matches_query, {'user_id': user_id})
            
            # Calculate expiry (tomorrow at 00:00)
            tomorrow = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            
            # Insert new recommendations
            for idx, rec in enumerate(recommendations):
                batch_num = (idx // settings.RECOMMENDATIONS_PER_REQUEST) + 1
                position = (idx % settings.RECOMMENDATIONS_PER_REQUEST) + 1
                
                cache_query = text("""
                    INSERT INTO recommendation_cache (
                        user_id, movie_id, score, recommendation_reason,
                        matched_user_ids, position, batch_number, expires_at
                    ) VALUES (
                        :user_id, :movie_id, :score, :reason,
                        :matched_users, :position, :batch_number, :expires_at
                    )
                """)
                
                self.db.execute(cache_query, {
                    'user_id': user_id,
                    'movie_id': rec.movie_id,
                    'score': rec.score,
                    'reason': rec.recommendation_reason,
                    'matched_users': rec.matched_user_ids,
                    'position': position,
                    'batch_number': batch_num,
                    'expires_at': tomorrow
                })
                
                # Cache potential matches
                for pm in rec.potential_matches:
                    pm_query = text("""
                        INSERT INTO potential_matches (
                            user_id, movie_id, potential_match_user_id,
                            match_condition, confidence_score, expires_at
                        ) VALUES (
                            :user_id, :movie_id, :match_user_id,
                            :condition, :confidence, :expires_at
                        )
                    """)
                    
                    self.db.execute(pm_query, {
                        'user_id': user_id,
                        'movie_id': rec.movie_id,
                        'match_user_id': pm.user_id,
                        'condition': pm.match_condition,
                        'confidence': pm.confidence_score,
                        'expires_at': tomorrow
                    })
            
            self.db.commit()
            logger.info(f"Cached {len(recommendations)} recommendations for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error caching recommendations for user {user_id}: {str(e)}")
            self.db.rollback()
            raise RecommendationEngineError(f"Failed to cache recommendations: {str(e)}")