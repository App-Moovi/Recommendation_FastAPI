from typing import List, Dict, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.utils.constants import MatchingThresholds, InteractionWeights
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class UserMatcher:
    """Handle user matching based on movie interactions"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_potential_matches(
        self, 
        user_id: int, 
        movie_id: int,
        already_matched_users: List[int] = None
    ) -> List[Dict]:
        """
        Get potential matches for a user based on a movie
        Returns list of {user_id, condition, confidence}
        """
        potential_matches = []
        already_matched = set(already_matched_users or [])
        
        # Get all existing matches for this user to exclude them
        existing_matches = self._get_existing_matches(user_id)
        excluded_users = already_matched.union(existing_matches)
        
        # Find users who have interacted with this movie
        interactions_query = text("""
            SELECT 
                user_id,
                rating,
                preference,
                CASE 
                    WHEN rating IS NOT NULL THEN 'rating'
                    ELSE 'preference'
                END as interaction_type
            FROM (
                SELECT user_id, rating, NULL as preference 
                FROM movie_ratings 
                WHERE movie_id = :movie_id AND user_id != :user_id
                UNION ALL
                SELECT user_id, NULL as rating, preference 
                FROM movie_preferences 
                WHERE movie_id = :movie_id AND user_id != :user_id
            ) interactions
        """)
        
        results = self.db.execute(
            interactions_query,
            {'movie_id': movie_id, 'user_id': user_id}
        ).fetchall()

        logger.debug(f'Found {len(results)} users who interacted with movie {movie_id}')
        
        # Group by match conditions
        condition_users = {
            'superlike': [],
            'like': [],
            'rating_5': [],
            'rating_4+': [],
            'rating_3+': [],
            'dislike': []
        }
        
        for row in results:
            other_user_id = row[0]
            if other_user_id in excluded_users:
                continue
            
            if row[3] == 'rating':
                rating = row[1]
                if rating == 5:
                    condition_users['rating_5'].append(other_user_id)
                elif rating == 4:
                    condition_users['rating_4+'].append(other_user_id)
                elif rating == 3:
                    condition_users['rating_3+'].append(other_user_id)
            else:
                preference = row[2]
                if preference == 'SUPERLIKE':
                    condition_users['superlike'].append(other_user_id)
                elif preference == 'LIKE':
                    condition_users['like'].append(other_user_id)
                elif preference == 'DISLIKE':
                    condition_users['dislike'].append(other_user_id)
        
        # For each condition, find the best match from that specific group
        for condition, users in condition_users.items():
            if not users:
                continue
            
            # Get similarity scores for these specific users
            best_match = self._find_best_match_from_group(user_id, users)
            
            if best_match and best_match[1] >= settings.MATCH_THRESHOLD:
                potential_matches.append({
                    'user_id': best_match[0],
                    'condition': condition,
                    'confidence': float(best_match[1])
                })
                
                logger.debug(f"Found potential match for condition {condition}: user {best_match[0]} with confidence {best_match[1]}")
        
        return potential_matches
    
    def _get_existing_matches(self, user_id: int) -> set:
        """Get all existing matches for a user"""
        matches_query = text("""
            SELECT 
                CASE 
                    WHEN user_id = :user_id THEN matched_user_id 
                    ELSE user_id 
                END as matched_user
            FROM matches
            WHERE :user_id IN (user_id, matched_user_id)
        """)
        
        results = self.db.execute(matches_query, {'user_id': user_id}).fetchall()
        return {row[0] for row in results}
    
    def _find_best_match_from_group(
        self, 
        user_id: int, 
        candidate_users: List[int]
    ) -> Tuple[int, float]:
        """Find the best matching user from a specific group of candidates"""
        if not candidate_users:
            return None
        
        # Get similarity scores for these specific candidates
        similarity_query = text("""
            SELECT 
                CASE 
                    WHEN user_id_1 = :user_id THEN user_id_2 
                    ELSE user_id_1 
                END as other_user_id,
                similarity_score
            FROM user_similarities
            WHERE (
                (user_id_1 = :user_id AND user_id_2 = ANY(:candidates))
                OR 
                (user_id_2 = :user_id AND user_id_1 = ANY(:candidates))
            )
            ORDER BY similarity_score DESC
        """)
        
        results = self.db.execute(
            similarity_query,
            {'user_id': user_id, 'candidates': candidate_users}
        ).fetchall()
        
        # Return the best match from this specific group
        if results:
            best_user_id = results[0][0]
            best_score = float(results[0][1])
            return (best_user_id, best_score)
        
        # If no pre-computed similarity, calculate on the fly for top candidate
        # (This should rarely happen if background jobs are running properly)
        return self._calculate_similarity_for_candidates(user_id, candidate_users[:3])
    
    def _calculate_similarity_for_candidates(
        self,
        user_id: int,
        candidate_users: List[int]
    ) -> Tuple[int, float]:
        """Calculate similarity on the fly for candidates"""
        # Get user interactions
        user_interactions = self._get_user_interactions(user_id)
        
        best_match = None
        best_score = 0
        
        # Check each candidate
        for candidate_id in candidate_users:
            candidate_interactions = self._get_user_interactions(candidate_id)
            
            # Calculate similarity
            common_movies = set(user_interactions.keys()) & set(candidate_interactions.keys())
            if len(common_movies) >= 2:  # Minimum 2 common movies
                # Simple cosine similarity
                score = self._calculate_cosine_similarity(
                    user_interactions,
                    candidate_interactions,
                    common_movies
                )
                
                if score > best_score:
                    best_score = score
                    best_match = candidate_id
        
        if best_match:
            return (best_match, best_score)
        
        return None
    
    def _get_user_interactions(self, user_id: int) -> Dict[int, float]:
        """Get all user interactions as weights"""
        interactions = {}
        
        # Get ratings
        rating_query = text("""
            SELECT movie_id, rating FROM movie_ratings WHERE user_id = :user_id
        """)
        for row in self.db.execute(rating_query, {'user_id': user_id}).fetchall():
            interactions[row[0]] = InteractionWeights.get_weight(None, row[1])
        
        # Get preferences
        pref_query = text("""
            SELECT movie_id, preference FROM movie_preferences WHERE user_id = :user_id
        """)
        for row in self.db.execute(pref_query, {'user_id': user_id}).fetchall():
            weight = InteractionWeights.get_weight(row[1])
            # Use higher weight if movie has both rating and preference
            if row[0] in interactions:
                interactions[row[0]] = max(interactions[row[0]], weight)
            else:
                interactions[row[0]] = weight
        
        return interactions
    
    def _calculate_cosine_similarity(
        self,
        user1_interactions: Dict[int, float],
        user2_interactions: Dict[int, float],
        common_movies: set
    ) -> float:
        """Calculate cosine similarity between two users"""
        import numpy as np
        
        if not common_movies:
            return 0.0
        
        vec1 = np.array([user1_interactions[m] for m in common_movies])
        vec2 = np.array([user2_interactions[m] for m in common_movies])
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = dot_product / (norm1 * norm2)
        
        # Normalize to 0-1 range
        return (cosine_sim + 1) / 2