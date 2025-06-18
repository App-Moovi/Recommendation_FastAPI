import numpy as np
from typing import Dict, List, Tuple, Set
from sqlalchemy.orm import Session
from app.utils.constants import InteractionWeights
import logging

logger = logging.getLogger(__name__)

class SimilarityCalculator:
    """Calculate similarities between users and movies"""
    
    @staticmethod
    def calculate_user_similarity(
        user1_interactions: Dict[int, float], 
        user2_interactions: Dict[int, float],
        movie_features: Dict[int, Dict] = None
    ) -> Tuple[float, int]:
        """
        Calculate similarity between two users based on their interactions
        Returns: (similarity_score, common_movies_count)
        """
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
        print(normalized_sim)
        
        # Apply penalty if users have very few common movies
        if len(common_movies) < 3:
            normalized_sim *= 0.7
        
        return float(normalized_sim), len(common_movies)
    
    @staticmethod
    def _calculate_content_based_user_similarity(
        user1_interactions: Dict[int, float],
        user2_interactions: Dict[int, float],
        movie_features: Dict[int, Dict]
    ) -> Tuple[float, int]:
        """Calculate similarity based on movie content features"""
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
    
    @staticmethod
    def calculate_movie_similarity(
        movie1_features: Dict,
        movie2_features: Dict
    ) -> float:
        """Calculate content-based similarity between two movies"""
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
            similarity_score += weights['popularity'] * pop_ratio
        
        return min(similarity_score, 1.0)

class RecommendationScorer:
    """Score movies for recommendations"""
    
    def __init__(self, db: Session):
        self.db = db
        self.similarity_calculator = SimilarityCalculator()
    
    def score_movie_for_user(
        self,
        user_id: int,
        movie_id: int,
        user_interactions: Dict[int, float],
        similar_users: List[Tuple[int, float]],
        movie_features: Dict,
        global_popularity: float
    ) -> Tuple[float, List[int], str]:
        """
        Score a movie for a user using hybrid approach
        Returns: (score, influencing_user_ids, reason)
        """
        scores = []
        reasons = []
        influencing_users = []
        
        # 1. Collaborative Filtering Score
        if similar_users:
            cf_score, cf_users = self._collaborative_score(
                movie_id, similar_users
            )
            if cf_score > 0:
                scores.append(('collaborative', cf_score * 0.5))
                influencing_users.extend(cf_users)
                reasons.append("Users with similar taste liked this")
        
        # 2. Content-Based Score
        cb_score = self._content_based_score(
            movie_id, user_interactions, movie_features
        )
        if cb_score > 0:
            scores.append(('content', cb_score * 0.3))
            reasons.append("Similar to movies you enjoyed")
        
        # 3. Popularity Score (for diversity and cold start)
        pop_score = min(global_popularity / 100, 1.0)  # Normalize popularity
        scores.append(('popularity', float(pop_score) * 0.2))
        
        # Calculate final score
        final_score = sum(score for _, score in scores)
        
        # Generate reason
        if not reasons:
            reasons.append("Popular movie in your preferred genres")
        
        reason = "; ".join(reasons[:2])  # Limit to 2 reasons
        
        return final_score, list(set(influencing_users)), reason
    
    def _collaborative_score(
        self,
        movie_id: int,
        similar_users: List[Tuple[int, float]]
    ) -> Tuple[float, List[int]]:
        """Calculate collaborative filtering score"""
        weighted_sum = 0
        similarity_sum = 0
        influencing_users = []
        
        # Get ratings from similar users
        for user_id, similarity in similar_users[:10]:  # Top 10 similar users
            # Check if similar user has interacted with this movie
            interaction = self._get_user_movie_interaction(user_id, movie_id)
            
            if interaction is not None:
                weighted_sum += similarity * interaction
                similarity_sum += similarity
                if interaction > 0:
                    influencing_users.append(user_id)
        
        if similarity_sum == 0:
            return 0.0, []
        
        score = weighted_sum / similarity_sum
        # Normalize to 0-1 range
        normalized_score = (score + 3) / 6  # Assuming weights range from -3 to 3
        
        return max(0, min(1, normalized_score)), influencing_users
    
    def _content_based_score(
        self,
        movie_id: int,
        user_interactions: Dict[int, float],
        movie_features: Dict
    ) -> float:
        """Calculate content-based score"""
        if movie_id not in movie_features:
            return 0.0
        
        target_features = movie_features[movie_id]
        similarity_scores = []
        
        # Compare with user's positively rated movies
        for liked_movie_id, interaction_score in user_interactions.items():
            if interaction_score > 0 and liked_movie_id != movie_id:
                if liked_movie_id in movie_features:
                    similarity = self.similarity_calculator.calculate_movie_similarity(
                        target_features,
                        movie_features[liked_movie_id]
                    )
                    # Weight by user's interaction score
                    weighted_sim = similarity * (interaction_score / 3)
                    similarity_scores.append(weighted_sim)
        
        if not similarity_scores:
            return 0.0
        
        # Return average of top 5 similar movies
        top_similarities = sorted(similarity_scores, reverse=True)[:5]
        return np.mean(top_similarities)
    
    def _get_user_movie_interaction(self, user_id: int, movie_id: int) -> float:
        """Get user's interaction weight for a movie"""
        # Check ratings first
        rating_query = f"""
            SELECT rating FROM movie_ratings 
            WHERE user_id = {user_id} AND movie_id = {movie_id}
        """
        result = self.db.execute(rating_query).fetchone()
        if result:
            return InteractionWeights.get_weight(None, result[0])
        
        # Check preferences
        pref_query = f"""
            SELECT preference FROM movie_preferences
            WHERE user_id = {user_id} AND movie_id = {movie_id}
        """
        result = self.db.execute(pref_query).fetchone()
        if result:
            return InteractionWeights.get_weight(result[0])
        
        return None