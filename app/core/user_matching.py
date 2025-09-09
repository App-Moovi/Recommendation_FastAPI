from typing import List, Dict, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import text, bindparam
from app.utils.constants import InteractionWeights
from app.config import settings
import logging
from app.utils.logger import timed
from collections import defaultdict
from app.models.schemas import PotentialMatch

logger = logging.getLogger(__name__)


class UserMatcher:
    """Handle user matching based on movie interactions"""

    def __init__(self, db: Session):
        self.db = db

    @timed
    def get_potential_matches(
        self,
        user_id: int,
        movie_ids: List[int],
        already_matched_users: List[int] = None,
    ) -> Dict[int, List[PotentialMatch]]:
        """
        Get potential matches for a user based on a movie
        Returns list of {user_id, condition, confidence}
        """
        already_matched = set(already_matched_users or [])

        # Find users who have interacted with these movies
        interactions_query = text("""
            SELECT 
                user_id,
                movie_id,
                rating,
                preference,
                CASE 
                    WHEN rating IS NOT NULL THEN 'rating'
                    ELSE 'preference'
                END as interaction_type
            FROM (
                SELECT user_id, movie_id, rating, NULL as preference 
                FROM movie_ratings 
                WHERE movie_id IN :movie_ids AND user_id != :user_id AND user_id NOT IN :excluded_users
                UNION ALL
                SELECT user_id, movie_id, NULL as rating, preference 
                FROM movie_preferences 
                WHERE movie_id IN :movie_ids AND user_id != :user_id AND user_id NOT IN :excluded_users
            ) interactions
        """).bindparams(
            bindparam("excluded_users", expanding=True),
            bindparam("movie_ids", expanding=True),
        )

        results = self.db.execute(
            interactions_query,
            {
                "movie_ids": movie_ids,
                "user_id": user_id,
                "excluded_users": list(already_matched),
            },
        ).fetchall()

        movie_lookup = defaultdict(dict)
        candidate_user_ids = set()

        for row in results:
            other_user_id = row[0]
            movie_id = row[1]
            rating = row[2]
            preference = row[3]
            interaction_type = row[4]

            if not other_user_id:
                continue

            candidate_user_ids.add(other_user_id)

            if movie_id not in movie_lookup:
                movie_lookup[movie_id] = {
                    "superlike": [],
                    "like": [],
                    "rating_5": [],
                    "rating_4+": [],
                    "rating_3+": [],
                    "dislike": [],
                }

            if interaction_type == "rating":
                if rating == 5:
                    movie_lookup[movie_id]["rating_5"].append(other_user_id)
                elif rating == 4:
                    movie_lookup[movie_id]["rating_4+"].append(other_user_id)
                elif rating == 3:
                    movie_lookup[movie_id]["rating_3+"].append(other_user_id)
            else:
                if preference == "SUPERLIKE":
                    movie_lookup[movie_id]["superlike"].append(other_user_id)
                elif preference == "LIKE":
                    movie_lookup[movie_id]["like"].append(other_user_id)
                elif preference == "DISLIKE":
                    movie_lookup[movie_id]["dislike"].append(other_user_id)

        logger.debug(
            f"Found {len(movie_lookup)} potential matches based on movies for user {user_id}"
        )

        interactions_by_user = self._get_interactions_batch(
            [user_id] + list(candidate_user_ids)
        )

        potential_matches: Dict[int, List[PotentialMatch]] = defaultdict(list)
        # For each condition, find the best match
        for movie_id, condition_users in movie_lookup.items():
            logger.info(f"Processing potential matches for movie {movie_id}")
            for condition, users in condition_users.items():
                logger.info(f"Processing potential matches for condition {condition}")
                logger.info(f"Found {len(users)} users for condition {condition}")

                if not users:
                    continue

                best_match = self._find_best_match_from_group(
                    user_id, users, interactions_by_user
                )

                if not best_match:
                    logger.info(
                        f"No best match found for movie {movie_id} and condition {condition}"
                    )
                    continue

                logger.info(
                    f"Found best match for movie {movie_id} and condition {condition}: user {best_match[0]} with confidence {best_match[1]}"
                )

                if best_match and best_match[1] >= settings.MATCH_THRESHOLD:
                    potential_match = PotentialMatch(
                        user_id=best_match[0],
                        match_condition=condition,
                        confidence_score=float(best_match[1]),
                    )

                    potential_matches[movie_id].append(potential_match)
                    logger.debug(
                        f"Found potential match for condition {condition}: user {best_match[0]} with confidence {best_match[1]}"
                    )

        return potential_matches

    def _find_best_match_from_group(
        self,
        user_id: int,
        candidate_users: List[int],
        interactions_by_user: Dict[int, Dict[int, float]],
    ) -> Tuple[int, float]:
        """Find best match within a group of users (uses pre-fetched interactions)"""
        user_interactions = interactions_by_user.get(user_id, {})
        if not user_interactions or not candidate_users:
            logger.error(f"Error in _find_best_match_from_group: user_interactions: {user_interactions}, candidate_users: {candidate_users}")
            return None

        best_match = None
        best_score = 0

        for candidate_id in candidate_users:
            candidate_interactions = interactions_by_user.get(candidate_id, {})
            if not candidate_interactions:
                continue

            common_movies = set(user_interactions) & set(candidate_interactions)
            if len(common_movies) < 2:
                continue

            if(not user_interactions or not candidate_interactions or not common_movies):
                logger.error(f"Error in _find_best_match_from_group: user_interactions: {user_interactions}, candidate_interactions: {candidate_interactions}, common_movies: {common_movies}")
                continue

            score = self._calculate_cosine_similarity(
                user_interactions, candidate_interactions, common_movies
            )

            if score > best_score:
                best_score = score
                best_match = candidate_id

        return (best_match, best_score) if best_match else None

    def _get_interactions_batch(
        self, user_ids: List[int]
    ) -> Dict[int, Dict[int, float]]:
        """Fetch all interactions for multiple users in one query"""
        interactions = {uid: {} for uid in user_ids}

        # Ratings
        rating_query = text("""
            SELECT user_id, movie_id, rating
            FROM movie_ratings
            WHERE user_id = ANY(:user_ids)
        """)
        for row in self.db.execute(rating_query, {"user_ids": user_ids}).fetchall():
            uid, mid, rating = row
            interactions[uid][mid] = InteractionWeights.get_weight(None, rating)

        # Preferences
        pref_query = text("""
            SELECT user_id, movie_id, preference
            FROM movie_preferences
            WHERE user_id = ANY(:user_ids)
        """)
        for row in self.db.execute(pref_query, {"user_ids": user_ids}).fetchall():
            uid, mid, pref = row
            weight = InteractionWeights.get_weight(pref)

            if mid in interactions[uid]:
                interactions[uid][mid] = max(interactions[uid][mid], weight)
            else:
                interactions[uid][mid] = weight

        return interactions

    @timed
    def _calculate_cosine_similarity(
        self,
        user1_interactions: Dict[int, float],
        user2_interactions: Dict[int, float],
        common_movies: set,
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
