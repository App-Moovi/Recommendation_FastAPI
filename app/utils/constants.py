from enum import Enum

class InteractionType(str, Enum):
    LIKE = "LIKE"
    DISLIKE = "DISLIKE"
    SUPERLIKE = "SUPERLIKE"
    SEEN = "SEEN"
    HIDE = "HIDE"

class InteractionWeights:
    """Weights for different interaction types"""
    SUPERLIKE = 3.0      # Equivalent to 5-star rating
    LIKE = 2.0           # Positive signal
    RATING_5 = 3.0       # Same as superlike
    RATING_4 = 2.0       # Strong positive
    RATING_3 = 1.0       # Moderate positive
    RATING_2 = -1.0      # Moderate negative
    RATING_1 = -2.0      # Strong negative (same as dislike)
    DISLIKE = -2.0       # Negative signal
    SEEN = 0.5           # Mild positive (user completed watching)
    HIDE = -3.0          # Strong negative signal
    
    @classmethod
    def get_weight(cls, interaction_type: str, rating: float = None) -> float:
        """Get weight for an interaction"""
        if rating is not None:
            weight_map = {
                5: cls.RATING_5,
                4: cls.RATING_4,
                3: cls.RATING_3,
                2: cls.RATING_2,
                1: cls.RATING_1
            }
            return weight_map.get(int(rating), 0)
        
        weight_map = {
            InteractionType.SUPERLIKE: cls.SUPERLIKE,
            InteractionType.LIKE: cls.LIKE,
            InteractionType.DISLIKE: cls.DISLIKE,
            InteractionType.SEEN: cls.SEEN,
            InteractionType.HIDE: cls.HIDE
        }
        return weight_map.get(interaction_type, 0)

class DiversitySettings:
    MAX_SAME_GENRE = 10  # Max movies from same genre in 20 recommendations
    MAX_SAME_YEAR = 10   # Max movies from same release year
    MAX_SAME_LANGUAGE = 20  # Max movies from same language
    
class MatchingThresholds:
    """Thresholds for matching users based on interactions"""
    SUPERLIKE_MATCH = True  # Always match on superlike
    LIKE_MATCH = True       # Always match on like
    DISLIKE_MATCH = True    # Match on shared dislikes
    RATING_MATCH_THRESHOLD = 3  # Match if rating >= 3