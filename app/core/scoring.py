import numpy as np
from typing import Dict, List, Tuple, Set
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.utils.constants import InteractionWeights
import logging
from app.utils.logger import timed

logger = logging.getLogger(__name__)

class ScoringError(Exception):
    """Custom exception for scoring errors"""
    pass

class SimilarityCalculator:
    """Calculate similarities between users and movies"""
    
    @staticmethod
    @timed
    def calculate_user_similarity(
        user1_interactions: Dict[int, float], 
        user2_interactions: Dict[int, float],
        movie_features: Dict[int, Dict] = None
    ) -> Tuple[float, int]:
        """
        Calculate similarity between two users based on their interactions
        Returns: (similarity_score, common_movies_count)
        """
        try:
            # Find common movies
            common_movies = set(user1_interactions.keys()) & set(user2_interactions.keys())
            
            if not common_movies:
                # If no common movies, check for similar movie preferences
                if movie_features:
                    return SimilarityCalculator._calculate_content_based_user_similarity(
                        user1_interactions, user2_interactions, movie_features
                    )
                return 0.0, 0
            
            # Calculate cosine similarity for common movies
            user1_scores = []
            user2_scores = []
            
            for movie_id in common_movies:
                user1_scores.append(user1_interactions[movie_id])
                user2_scores.append(user2_interactions[movie_id])
            
            # Convert to numpy arrays
            user1_vec = np.array(user1_scores)
            user2_vec = np.array(user2_scores)
            
            # Calculate cosine similarity
            dot_product = np.dot(user1_vec, user2_vec)
            norm1 = np.linalg.norm(user1_vec)
            norm2 = np.linalg.norm(user2_vec)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0, len(common_movies)
            
            cosine_sim = dot_product / (norm1 * norm2)
            
            # Normalize to 0-1 range
            normalized_sim = (cosine_sim + 1) / 2
            
            # Boost similarity if users have many common movies
            if len(common_movies) >= 10:
                normalized_sim = min(1.0, normalized_sim * 1.2)
            elif len(common_movies) < 3:
                normalized_sim *= 0.7
            
            logger.debug(f"User similarity calculated: {normalized_sim:.3f} with {len(common_movies)} common movies")
            
            return float(normalized_sim), len(common_movies)
            
        except Exception as e:
            logger.error(f"Error calculating user similarity: {str(e)}")
            return 0.0, 0
    
    @staticmethod
    @timed
    def _calculate_content_based_user_similarity(
        user1_interactions: Dict[int, float],
        user2_interactions: Dict[int, float],
        movie_features: Dict[int, Dict]
    ) -> Tuple[float, int]:
        """Calculate similarity based on movie content features"""
        try:
            # Extract preferred genres for each user
            user1_genres = set()
            user2_genres = set()
            
            for movie_id, score in user1_interactions.items():
                if score > 0 and movie_id in movie_features:
                    genres = movie_features[movie_id].get('genres', [])
                    user1_genres.update(genres)
            
            for movie_id, score in user2_interactions.items():
                if score > 0 and movie_id in movie_features:
                    genres = movie_features[movie_id].get('genres', [])
                    user2_genres.update(genres)
            
            if not user1_genres or not user2_genres:
                return 0.0, 0
            
            # Jaccard similarity for genres
            intersection = len(user1_genres & user2_genres)
            union = len(user1_genres | user2_genres)
            
            if union == 0:
                return 0.0, 0
            
            similarity = intersection / union
            return similarity * 0.5, 0  # Reduce weight for content-only similarity
            
        except Exception as e:
            logger.error(f"Error calculating content-based user similarity: {str(e)}")
            return 0.0, 0
    
    @staticmethod
    @timed
    def calculate_movie_similarity(
        movie1_features: Dict,
        movie2_features: Dict
    ) -> float:
        """Calculate content-based similarity between two movies"""
        try:
            similarity_score = 0.0
            weights = {
                'genres': 0.4,
                'cast': 0.2,
                'production_companies': 0.1,
                'language': 0.1,
                'year_diff': 0.1,
                'popularity': 0.1
            }
            
            # Genre similarity (Jaccard)
            genres1 = set(movie1_features.get('genres', []))
            genres2 = set(movie2_features.get('genres', []))
            if genres1 and genres2:
                genre_sim = len(genres1 & genres2) / len(genres1 | genres2)
                similarity_score += weights['genres'] * genre_sim
            
            # Cast similarity
            cast1 = set(movie1_features.get('cast_ids', []))
            cast2 = set(movie2_features.get('cast_ids', []))
            if cast1 and cast2:
                cast_sim = len(cast1 & cast2) / min(len(cast1), len(cast2), 5)  # Consider top 5 cast
                similarity_score += weights['cast'] * min(cast_sim, 1.0)
            
            # Production company similarity
            prod1 = set(movie1_features.get('production_companies', []))
            prod2 = set(movie2_features.get('production_companies', []))
            if prod1 and prod2:
                prod_sim = len(prod1 & prod2) / len(prod1 | prod2)
                similarity_score += weights['production_companies'] * prod_sim
            
            # Language similarity
            if movie1_features.get('language') == movie2_features.get('language'):
                similarity_score += weights['language']
            
            # Year difference (movies from similar era)
            year1 = movie1_features.get('year', 2020)
            year2 = movie2_features.get('year', 2020)
            year_diff = abs(year1 - year2)
            year_sim = max(0, 1 - (year_diff / 10))  # 10 years = 0 similarity
            similarity_score += weights['year_diff'] * year_sim
            
            # Popularity similarity
            pop1 = movie1_features.get('popularity', 0)
            pop2 = movie2_features.get('popularity', 0)
            if pop1 > 0 and pop2 > 0:
                pop_ratio = min(pop1, pop2) / max(pop1, pop2)
                similarity_score += weights['popularity'] * float(pop_ratio)
            
            return min(similarity_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating movie similarity: {str(e)}")
            return 0.0

class RecommendationScorer:
    """Score movies for recommendations with improved scoring system"""
    
    def __init__(self, db: Session):
        self.db = db
        self.similarity_calculator = SimilarityCalculator()
        self._movie_features_cache = {}  # Cache for movie features

    @timed    
    def score_movie_for_user(
        self,
        user_id: int,
        movie_id: int,
        user_interactions: Dict[int, float],
        similar_users: List[Tuple[int, float]],
        movie_features: Dict,
        global_popularity: float,
        user_genres: List[int] = None,
        weights: Dict[str, float] = None
    ) -> Tuple[float, List[int], str]:
        """
        Score a movie for a user using hybrid approach with improved scoring
        Returns: (score, influencing_user_ids, reason)
        """
        try:
            # Improved adaptive weights based on user's state
            if weights is None:
                has_interactions = len(user_interactions) > 0
                has_similar_users = len(similar_users) > 0
                has_genres = user_genres and len(user_genres) > 0
                
                if not has_interactions and not has_similar_users:
                    # Pure cold start - rely heavily on genres and popularity
                    weights = {
                        'collaborative': 0.0,
                        'content': 0.0,
                        'genre': 0.7,      # Very high weight on user's preferred genres
                        'popularity': 0.3,
                        'quality': 0.2,    # New: movie quality score
                        'trending': 0.1    # New: trending boost
                    }
                elif has_interactions < 10:  # Few interactions
                    weights = {
                        'collaborative': 0.15,
                        'content': 0.25,
                        'genre': 0.35,     # Still significant genre weight
                        'popularity': 0.15,
                        'quality': 0.1,
                        'trending': 0.0
                    }
                else:  # Normal case with enough data
                    weights = {
                        'collaborative': 0.2,
                        'content': 0.3,
                        'genre': 0.25,     # Balanced genre weight
                        'popularity': 0.1,
                        'quality': 0.1,
                        'trending': 0.05
                    }
            
            scores = []
            reasons = []
            influencing_users = []
            
            # 1. Collaborative Filtering Score
            if similar_users and weights.get('collaborative', 0) > 0:
                cf_score, cf_users = self._collaborative_score(
                    movie_id, similar_users
                )
                if cf_score > 0:
                    scores.append(('collaborative', cf_score * weights['collaborative']))
                    influencing_users.extend(cf_users)
                    reasons.append("Users with similar taste loved this")
            
            # 2. Content-Based Score
            if len(user_interactions) > 0 and weights.get('content', 0) > 0:
                # Fetch features for user's interacted movies for content-based scoring
                user_movie_features = self._get_user_movie_features(user_interactions)
                cb_score = self._content_based_score(
                    movie_id, user_interactions, movie_features, user_movie_features
                )
                
                if cb_score > 0:
                    scores.append(('content', cb_score * weights['content']))
                    if cb_score > 0.7:
                        reasons.append("Very similar to movies you enjoyed")
                    else:
                        reasons.append("Similar to movies you enjoyed")
                
                logger.debug(f'Content-based score for movie {movie_id}: {cb_score:.3f}')
            
            # 3. Genre-Based Score (improved scoring)
            if user_genres and weights.get('genre', 0) > 0:
                genre_score = self._genre_based_score(movie_features, user_genres)
                if genre_score > 0:
                    # Boost genre score for better differentiation
                    boosted_genre_score = min(1.0, genre_score * 1.3)
                    scores.append(('genre', boosted_genre_score * weights['genre']))
                    if genre_score > 0.8:
                        reasons.append("Perfect match for your favorite genres")
                    elif genre_score > 0.5:
                        reasons.append("Matches your favorite genres")
            
            # 4. Popularity Score (normalized better)
            if weights.get('popularity', 0) > 0:
                # Use logarithmic scale for popularity to avoid extreme values
                # Convert Decimal to float first
                popularity_float = float(global_popularity) if global_popularity else 0.0
                pop_score = min(np.log1p(popularity_float) / 10, 1.0)
                scores.append(('popularity', pop_score * weights['popularity']))
            
            # 5. Quality Score (new component)
            if weights.get('quality', 0) > 0:
                quality_score = self._calculate_quality_score(movie_features)
                scores.append(('quality', quality_score * weights['quality']))
                if quality_score > 0.8:
                    reasons.append("Highly rated by critics")
            
            # 6. Trending Score (new component)
            if weights.get('trending', 0) > 0:
                trending_score = self._get_trending_score(movie_id)
                if trending_score > 0:
                    scores.append(('trending', trending_score * weights['trending']))
                    reasons.append("Currently trending")
            
            # Calculate final score with bonus multipliers
            base_score = sum(score for _, score in scores)
            
            # Apply bonus multipliers for exceptional matches
            final_score = base_score
            
            # Genre match bonus
            if user_genres and len(set(movie_features.get('genres', [])) & set(user_genres)) >= 2:
                final_score *= 1.15  # 15% bonus for multi-genre match
            
            # High quality bonus
            if movie_features.get('vote_average', 0) >= 7.5 and movie_features.get('vote_count', 0) >= 100:
                final_score *= 1.1   # 10% bonus for high quality
            
            # Cap the score at 1.0
            final_score = min(final_score, 1.0)
            
            # Log component scores for debugging
            logger.debug(f"Movie {movie_id} scoring breakdown: " + 
                        ", ".join([f"{name}: {score:.3f}" for name, score in scores]) +
                        f", final: {final_score:.3f}")
            
            # Generate reason
            if not reasons:
                if user_genres and len(set(movie_features.get('genres', [])) & set(user_genres)) > 0:
                    reasons.append("Popular movie in your preferred genres")
                else:
                    reasons.append("Popular movie")
            
            reason = "; ".join(reasons[:2])  # Limit to 2 reasons
            
            return final_score, list(set(influencing_users)), reason
            
        except Exception as e:
            logger.error(f"Error scoring movie {movie_id} for user {user_id}: {str(e)}")
            raise ScoringError(f"Failed to score movie: {str(e)}")
    
    @timed
    def _collaborative_score(
        self,
        movie_id: int,
        similar_users: List[Tuple[int, float]]
    ) -> Tuple[float, List[int]]:
        """Calculate collaborative filtering score"""
        try:
            weighted_sum = 0
            similarity_sum = 0
            influencing_users = []
            
            # Get ratings from similar users
            for user_id, similarity in similar_users[:10]:  # Top 10 similar users
                # Check if similar user has interacted with this movie
                interaction = self._get_user_movie_interaction(user_id, movie_id)
                
                if interaction is not None:
                    # Ensure similarity is float (might be Decimal from DB)
                    similarity_float = float(similarity)
                    interaction_float = float(interaction)
                    weighted_sum += similarity_float * interaction_float
                    similarity_sum += similarity_float
                    if interaction > 0:
                        influencing_users.append(user_id)
            
            if similarity_sum == 0:
                return 0.0, []
            
            score = weighted_sum / similarity_sum
            # Normalize to 0-1 range (assuming weights range from -3 to 3)
            normalized_score = (score + 3) / 6
            
            return max(0, min(1, normalized_score)), influencing_users
            
        except Exception as e:
            logger.error(f"Error calculating collaborative score: {str(e)}")
            return 0.0, []
    
    @timed
    def _content_based_score(
        self,
        movie_id: int,
        user_interactions: Dict[int, float],
        target_movie_features: Dict,
        user_movie_features: Dict[int, Dict]
    ) -> float:
        """Calculate content-based score"""
        try:
            if not target_movie_features:
                return 0.0
            
            similarity_scores = []
            
            # Compare with user's positively rated movies
            for liked_movie_id, interaction_score in user_interactions.items():
                if interaction_score > 0 and liked_movie_id != movie_id:
                    if liked_movie_id in user_movie_features:
                        similarity = self.similarity_calculator.calculate_movie_similarity(
                            target_movie_features,
                            user_movie_features[liked_movie_id]
                        )
                        # Weight by user's interaction score
                        weighted_sim = similarity * (interaction_score / 3)
                        similarity_scores.append(weighted_sim)
            
            if not similarity_scores:
                return 0.0
            
            # Return weighted average of top similar movies
            top_similarities = sorted(similarity_scores, reverse=True)[:10]
            
            # Use weighted average with decreasing weights for lower similarities
            weighted_avg = 0
            weight_sum = 0
            for i, sim in enumerate(top_similarities):
                weight = 1.0 / (i + 1)  # Decreasing weights: 1, 0.5, 0.33, 0.25...
                weighted_avg += sim * weight
                weight_sum += weight
            
            return weighted_avg / weight_sum if weight_sum > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating content-based score: {str(e)}")
            return 0.0
    
    @timed
    def _genre_based_score(self, movie_features: Dict, user_genres: List[int]) -> float:
        """
        Calculate genre-based score for a movie with improved scoring
        """
        try:
            if not movie_features or not user_genres:
                return 0.0
            
            movie_genres = set(movie_features.get('genres', []))
            user_genre_set = set(user_genres)
            
            if not movie_genres:
                return 0.0
            
            # Calculate genre overlap
            common_genres = movie_genres & user_genre_set
            
            if not common_genres:
                return 0.0
            
            # Improved scoring based on:
            # 1. Number of matching genres
            # 2. Proportion of user's genres that match
            # 3. Proportion of movie's genres that match
            
            # Base score: proportion of user's genres found in movie
            user_genre_coverage = len(common_genres) / len(user_genre_set) if user_genre_set else 0
            
            # Bonus for movies that are focused on user's genres
            movie_genre_focus = len(common_genres) / len(movie_genres) if movie_genres else 0
            
            # Combined score with emphasis on user coverage
            base_score = (0.7 * user_genre_coverage + 0.3 * movie_genre_focus)
            
            # Apply bonuses for multiple genre matches
            if len(common_genres) >= 3:
                base_score = min(1.0, base_score * 1.3)  # 30% bonus for 3+ genres
            elif len(common_genres) >= 2:
                base_score = min(1.0, base_score * 1.15)  # 15% bonus for 2 genres
            
            logger.debug(f"Genre score: {base_score:.3f}, common genres: {common_genres}")
            
            return base_score
            
        except Exception as e:
            logger.error(f"Error calculating genre-based score: {str(e)}")
            return 0.0
    
    def _calculate_quality_score(self, movie_features: Dict) -> float:
        """Calculate quality score based on ratings and vote count"""
        try:
            vote_average = movie_features.get('vote_average', 0)
            vote_count = movie_features.get('vote_count', 0)
            
            if vote_count < 10:  # Too few votes
                return 0.0
            
            # Normalize vote average (0-10 scale to 0-1)
            normalized_rating = vote_average / 10
            
            # Apply confidence based on vote count (sigmoid function)
            confidence = 1 - np.exp(-vote_count / 100)
            
            # Quality score is rating weighted by confidence
            quality_score = float(normalized_rating) * float(confidence)
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {str(e)}")
            return 0.0
    
    def _get_trending_score(self, movie_id: int) -> float:
        """Get trending score for a movie"""
        try:
            # Check if movie is in trending list
            trending_query = text("""
                SELECT 1 FROM movie_list_trending 
                WHERE movie_id = :movie_id
                LIMIT 1
            """)
            
            result = self.db.execute(trending_query, {'movie_id': movie_id}).fetchone()
            
            return 0.5 if result else 0.0  # Fixed score for trending movies
            
        except Exception as e:
            logger.error(f"Error getting trending score: {str(e)}")
            return 0.0
    
    def _get_user_movie_features(self, user_interactions: Dict[int, float]) -> Dict[int, Dict]:
        """Get features for all movies the user has interacted with"""
        try:
            # Only fetch features for positively interacted movies
            movie_ids = [movie_id for movie_id, score in user_interactions.items() if score > 0]
            
            # Check cache first
            uncached_ids = [mid for mid in movie_ids if mid not in self._movie_features_cache]
            
            if uncached_ids:
                # Batch fetch features for uncached movies
                features = self._batch_get_movie_features(uncached_ids)
                self._movie_features_cache.update(features)
            
            # Return features for requested movies
            return {mid: self._movie_features_cache[mid] for mid in movie_ids if mid in self._movie_features_cache}
            
        except Exception as e:
            logger.error(f"Error getting user movie features: {str(e)}")
            return {}
    
    def _batch_get_movie_features(self, movie_ids: List[int]) -> Dict[int, Dict]:
        """Batch fetch movie features for efficiency"""
        try:
            if not movie_ids:
                return {}
            
            features_dict = {}
            
            # Get basic movie info for all movies at once
            movie_query = text("""
                SELECT 
                    m.id, m.title, m.overview, m.genre, m.popularity,
                    m.vote_average, m.release_date, m.original_language,
                    m.runtime, m.poster_path, m.backdrop_path, m.vote_count
                FROM movies m
                WHERE m.id = ANY(:movie_ids)
            """)
            
            movie_results = self.db.execute(movie_query, {'movie_ids': movie_ids}).fetchall()
            
            for row in movie_results:
                movie_id = row[0]
                features_dict[movie_id] = {
                    'movie_id': movie_id,
                    'title': row[1],
                    'overview': row[2],
                    'genre': row[3],
                    'popularity': row[4] or 0,
                    'vote_average': row[5] or 0,
                    'release_date': row[6],
                    'language': row[7],
                    'runtime': row[8],
                    'poster_path': row[9],
                    'backdrop_path': row[10],
                    'vote_count': row[11] or 0,
                    'year': row[6].year if row[6] else None,
                    'genres': [],
                    'cast_ids': [],
                    'production_companies': []
                }
            
            # Batch get genres
            genre_query = text("""
                SELECT movie_id, genre_id 
                FROM movie_list_genres 
                WHERE movie_id = ANY(:movie_ids)
            """)
            genre_results = self.db.execute(genre_query, {'movie_ids': movie_ids}).fetchall()
            
            for movie_id, genre_id in genre_results:
                if movie_id in features_dict:
                    features_dict[movie_id]['genres'].append(genre_id)
            
            # Batch get cast (top 5 per movie)
            cast_query = text("""
                SELECT DISTINCT ON (mcr.movie_id, mc.popularity) 
                    mcr.movie_id, mcr.cast_id, mc.popularity
                FROM movie_cast_relations mcr
                JOIN movie_cast mc ON mcr.cast_id = mc.id
                WHERE mcr.movie_id = ANY(:movie_ids)
                ORDER BY mcr.movie_id, mc.popularity DESC
            """)
            cast_results = self.db.execute(cast_query, {'movie_ids': movie_ids}).fetchall()
            
            # Group cast by movie and take top 5
            from collections import defaultdict
            cast_by_movie = defaultdict(list)
            for movie_id, cast_id, _ in cast_results:
                cast_by_movie[movie_id].append(cast_id)
            
            for movie_id, cast_list in cast_by_movie.items():
                if movie_id in features_dict:
                    features_dict[movie_id]['cast_ids'] = cast_list[:5]
            
            # Batch get production companies
            prod_query = text("""
                SELECT movie_id, production_company_id 
                FROM production_company_relations 
                WHERE movie_id = ANY(:movie_ids)
            """)
            prod_results = self.db.execute(prod_query, {'movie_ids': movie_ids}).fetchall()
            
            for movie_id, prod_id in prod_results:
                if movie_id in features_dict:
                    features_dict[movie_id]['production_companies'].append(prod_id)
            
            return features_dict
            
        except Exception as e:
            logger.error(f"Error batch getting movie features: {str(e)}")
            return {}
    
    def _get_user_movie_interaction(self, user_id: int, movie_id: int) -> float:
        """Get user's interaction weight for a movie"""
        try:
            # Check ratings first
            rating_query = text("""
                SELECT rating FROM movie_ratings 
                WHERE user_id = :user_id AND movie_id = :movie_id
            """)
            result = self.db.execute(rating_query, {'user_id': user_id, 'movie_id': movie_id}).fetchone()
            if result:
                return InteractionWeights.get_weight(None, result[0])
            
            # Check preferences
            pref_query = text("""
                SELECT preference FROM movie_preferences
                WHERE user_id = :user_id AND movie_id = :movie_id
            """)
            result = self.db.execute(pref_query, {'user_id': user_id, 'movie_id': movie_id}).fetchone()
            if result:
                return InteractionWeights.get_weight(result[0])
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting user movie interaction: {str(e)}")
            return None