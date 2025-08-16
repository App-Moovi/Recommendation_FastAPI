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
from app.utils.common_tasks import commonTasks
from app.models.database_models import MovieSimilarities
from collections import defaultdict

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
            logger.info(f"Generating recommendations in recommendation engine for user {user_id}")
            # Check if we need to use cached recommendations
            if not force_refresh:
                cached = self._get_cached_recommendations(user_id)
                if cached and len(cached) >= count:
                    logger.info(f"Returning {count} cached recommendations for user {user_id}")
                    return cached[:count]
            
            # Get user profile using materialized view
            user_profile = self._get_user_profile(user_id)
            
            if not user_profile:
                raise RecommendationEngineError(f"User profile not found for user_id: {user_id}")
            
            # Get candidate movies
            candidate_movies = self._get_candidate_movies(user_id, user_profile)
            
            if not candidate_movies:
                raise RecommendationEngineError(f"No candidate movies found for user {user_id}")
            
            logger.info(f"Found {len(candidate_movies)} candidate movies for user {user_id}")
            
            # Score all candidates
            scored_movies = []
            all_movie_features = commonTasks.get_movies_features(list(candidate_movies), db=self.db)
            interacted_movie_combinations = set()
            for candidate in candidate_movies:
                for interaction in user_profile['interactions'].keys():
                    interacted_movie_combinations.add((candidate, interaction))

            interacted_movie_features = commonTasks.get_movies_features(list(user_profile.get('interactions', {}).keys()), db=self.db)
            interaction_lookup_table = defaultdict(dict)

            for movie_features in interacted_movie_features:
                movie_id = movie_features.movie_id
                interaction_score = user_profile.get('interactions', {}).get(movie_id, 0)
                if not interaction_score:
                    continue

                interaction_lookup_table[movie_id] = {
                    'features': movie_features,
                    'interaction_score': interaction_score
                }

            for movie_features in all_movie_features:
                movie_id = movie_features.movie_id

                try:
                    # Use improved scoring with better weights
                    score, influencing_users, reason = self.scorer.score_movie_for_user(
                        user_id=user_id,
                        movie_features=movie_features,
                        user_interactions=user_profile['interactions'],
                        similar_users=user_profile['similar_users'],
                        interacted_movie_features=interaction_lookup_table,
                        user_genres=user_profile['genres'],
                        weights=weights
                    )
                    
                    # TODO - Implement user matching
                    # # Get potential matches for this movie
                    # potential_matches = self.user_matcher.get_potential_matches(
                    #     user_id, movie_id, influencing_users
                    # )
                    
                    scored_movies.append({
                        'movie_id': movie_id,
                        'score': score,
                        'reason': reason,
                        'influencing_users': influencing_users,
                        'potential_matches': [], # NOTE - potential_matches
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
    
    def _get_user_profile(self, user_id: int) -> Dict:
        """Get comprehensive user profile using materialized view for better performance"""
        recentPreferenceCount = 50
        recentRatingCount = 30

        logger.info(f"Generating user profile for user {user_id}")

        try:
            profile = {
                'user_id': user_id,
                'interactions': {},
                'genres': [],
                'similar_users': [],
                'language_preferences': [],
                'stats': {}
            }

            preference_query = text("""
                SELECT count(*), preference
                FROM movie_preferences
                WHERE user_id = :user_id
                GROUP BY preference           
            """)
            preference_result = self.db.execute(preference_query, {'user_id': user_id}).fetchall()
            if preference_result:
                total_preferences = 0
                for count, preference in preference_result:
                    total_preferences += count
                    if preference == 'LIKE':
                        profile['stats']['liked_movies'] = count or 0
                    elif preference == 'SUPERLIKE':
                        profile['stats']['superliked_movies'] = count or 0
                    elif preference == 'DISLIKE':
                        profile['stats']['disliked_movies'] = count or 0

                profile['stats']['total_preferences'] = total_preferences

            rating_count_query = text("""
                SELECT count(*) as total_ratings, avg(rating) as avg_rating
                FROM movie_ratings
                WHERE user_id = :user_id
            """)
            rating_count_result = self.db.execute(rating_count_query, {'user_id': user_id}).fetchone()
            if rating_count_result:
                profile['stats']['total_ratings'] = rating_count_result.total_ratings or 0
                profile['stats']['avg_rating'] = rating_count_result.avg_rating or 0

            # Get Preferred Genres
            preferred_genres_query = text("""
                SELECT genre_id
                FROM user_genres
                WHERE user_id = :user_id
            """)
            preferred_genres_result = self.db.execute(preferred_genres_query, {'user_id': user_id}).fetchall()
            if preferred_genres_result:
                profile['genres'] = [genre_id for genre_id, in preferred_genres_result]

            # Get similar users
            similar_users_query = text("""
                SELECT matched_user_id
                FROM matches
                WHERE user_id = :user_id
            """)
            similar_users_result = self.db.execute(similar_users_query, {'user_id': user_id}).fetchall()
            if similar_users_result:
                profile['similar_users'] = [user_id for user_id, in similar_users_result]

            # Get language preferences
            language_preferences_query = text("""
                SELECT language_code
                FROM user_languages
                WHERE user_id = :user_id
            """)
            language_preferences_result = self.db.execute(language_preferences_query, {'user_id': user_id}).fetchall()
            if language_preferences_result:
                profile['language_preferences'] = [language_code for language_code, in language_preferences_result]
            
            # Get detailed user interactions
            interactions_query = text("""
                SELECT movie_id, rating, NULL as preference
                FROM (
                    SELECT movie_id, rating
                    FROM movie_ratings
                    WHERE user_id = :user_id
                ) r

                UNION ALL

                SELECT movie_id, NULL as rating, preference
                FROM (
                    SELECT movie_id, preference
                    FROM movie_preferences
                    WHERE user_id = :user_id
                ) p;
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
            
            
            logger.info(f"User {user_id} Profile: {profile}")
            return profile
            
        except Exception as e:
            logger.error(f"Error getting user profile for user {user_id}: {str(e)}")
            raise RecommendationEngineError(f"Failed to get user profile: {str(e)}")
    
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
                limit = 150 if is_cold_start else 100
                
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
                interaction_limit = 50

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
                    LIMIT :limit
                """).bindparams(bindparam("interacted_movies", expanding=True))
                
                # Get user's liked movies (positive interactions)
                liked_movies = [movie_id for movie_id, score in user_profile['interactions'].items() if score > 1.0]
                
                if liked_movies:
                    results = self.db.execute(
                        similar_movies_query,
                        {
                            'liked_movies': liked_movies[:50],  # Top 50 liked movies
                            'interacted_movies': list(interacted_movies) or [-1],
                            'limit': interaction_limit
                        }
                    ).fetchall()
                    
                    candidates.update(row[0] for row in results)
                    logger.info(f"Added {len(results)} movies from movie similarities")
            
            # 3. Movies from similar users (if not cold start)
            if user_profile['similar_users'] and not is_cold_start:
                similar_limit = 50

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
                    LIMIT :limit
                """).bindparams(bindparam("interacted_movies", expanding=True))
                
                similar_user_ids = [u[0] for u in user_profile['similar_users']]
                results = self.db.execute(
                    similar_users_movies_query,
                    {
                        'user_ids': similar_user_ids,
                        'interacted_movies': list(interacted_movies) or [-1],
                        'limit': similar_limit
                    }
                ).fetchall()
                
                candidates.update(row[0] for row in results)
                logger.info(f"Added {len(results)} movies from similar users")
            
            
            # 4. Trending movies (smaller portion)
            trending_limit = 40
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