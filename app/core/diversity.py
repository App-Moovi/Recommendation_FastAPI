from typing import List, Dict
from collections import defaultdict
from app.utils.constants import DiversitySettings
import random
from app.models.schemas import MovieFeatures, ScoredMovie


class DiversityOptimizer:
    """Ensure diversity in recommendations"""

    def optimize_diversity(
        self, scored_movies: List[ScoredMovie], target_count: int
    ) -> List[ScoredMovie]:
        """
        Optimize movie list for diversity while maintaining quality
        """

        if len(scored_movies) <= target_count:
            return scored_movies

        selected: List[ScoredMovie] = []
        genre_counts = defaultdict(int)
        year_counts = defaultdict(int)
        language_counts = defaultdict(int)

        # Group movies by score ranges for quality maintenance
        score_groups = self._group_by_score_ranges(scored_movies)

        for score_range in sorted(score_groups.keys(), reverse=True):
            candidates = score_groups[score_range]
            random.shuffle(candidates)  # Add some randomness within same score range

            for movie in candidates:
                if len(selected) >= target_count:
                    break

                # Check diversity constraints
                if self._passes_diversity_check(
                    movie, genre_counts, year_counts, language_counts
                ):
                    selected.append(movie)

                    # Update counts
                    movie_features: MovieFeatures = movie.features
                    genres = movie_features.genres
                    for genre in genres:
                        genre_counts[genre] += 1

                    year = movie_features.year
                    if year:
                        year_counts[year] += 1

                    language = movie_features.language
                    if language:
                        language_counts[language] += 1

            if len(selected) >= target_count:
                break

        # If we still need more movies, relax constraints
        if len(selected) < target_count:
            remaining = [m for m in scored_movies if m not in selected]
            remaining.sort(key=lambda x: x.score, reverse=True)
            selected.extend(remaining[: target_count - len(selected)])

        return selected[:target_count]

    def _group_by_score_ranges(
        self, movies: List[ScoredMovie]
    ) -> Dict[float, List[ScoredMovie]]:
        """Group movies by score ranges to maintain quality"""
        groups = defaultdict(list)

        # Create buckets of 0.1 score range
        for movie in movies:
            bucket = round(movie.score * 10) / 10
            groups[bucket].append(movie)

        return groups

    def _passes_diversity_check(
        self,
        movie: ScoredMovie,
        genre_counts: Dict[int, int],
        year_counts: Dict[int, int],
        language_counts: Dict[str, int],
    ) -> bool:
        """Check if adding this movie maintains diversity"""
        movie_features: MovieFeatures = movie.features

        # Check genre diversity
        genres = movie_features.genres
        for genre in genres:
            if genre_counts.get(genre, 0) >= DiversitySettings.MAX_SAME_GENRE:
                return False

        # Check year diversity
        year = movie_features.year
        if year and year_counts.get(year, 0) >= DiversitySettings.MAX_SAME_YEAR:
            return False

        # Check language diversity
        language = movie_features.language
        if (
            language
            and language_counts.get(language, 0) >= DiversitySettings.MAX_SAME_LANGUAGE
        ):
            return False

        return True
