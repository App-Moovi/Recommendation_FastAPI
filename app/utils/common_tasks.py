from typing import List, Dict, Optional
import logging
from sqlalchemy import text, bindparam
from app.database import SessionLocal
from sqlalchemy.orm import Session
from collections import defaultdict
from app.models.schemas import MovieFeatures

logger = logging.getLogger(__name__)

class CommonTasksError(Exception):
    pass

class CommonTasks:

    @staticmethod
    def get_movies_features(movie_ids: List[int], db: Optional[Session] = None) -> List[MovieFeatures]:
        db = SessionLocal() if db is None else db

        try:
            movies_query = text("""
                SELECT id, title, overview, genre, popularity, vote_average, release_date, original_language, runtime, vote_count
                FROM movies
                WHERE id IN :movie_ids;
            """).bindparams(bindparam("movie_ids", expanding=True))

            cast_query = text("""
                SELECT movie_id, Array_Agg(cast_id)
                FROM movie_cast_relations
                WHERE movie_id IN :movie_ids
                GROUP BY movie_id;
            """).bindparams(bindparam("movie_ids", expanding=True))

            prod_query = text("""
                SELECT movie_id, Array_Agg(production_company_id)
                FROM production_company_relations
                WHERE movie_id IN :movie_ids
                GROUP BY movie_id;               
            """).bindparams(bindparam("movie_ids", expanding=True))

            genre_query = text("""
                SELECT movie_id, Array_Agg(genre_id)
                FROM movie_genre_links
                WHERE movie_id IN :movie_ids
                GROUP BY movie_id;
            """).bindparams(bindparam("movie_ids", expanding=True))

            movies = db.execute(movies_query, {'movie_ids': movie_ids}).fetchall()
            casts = db.execute(cast_query, {'movie_ids': movie_ids}).fetchall()
            prods = db.execute(prod_query, {'movie_ids': movie_ids}).fetchall()
            genres = db.execute(genre_query, {'movie_ids': movie_ids}).fetchall()

            lookup_table: Dict[int, Dict] = defaultdict(dict)

            movieLen, castLen, prodLen, genreLen = len(movies), len(casts), len(prods), len(genres)
            maxLen = max(movieLen, castLen, prodLen, genreLen)
            for i in range(maxLen):
                movie = movies[i] if i < movieLen else None
                cast = casts[i] if i < castLen else None
                prod = prods[i] if i < prodLen else None
                genre = genres[i] if i < genreLen else None

                if movie is not None:
                    lookup_table[movie[0]]['movie'] = {
                        'id': movie[0],
                        'title': movie[1],
                        'overview': movie[2],
                        'genre': movie[3],
                        'popularity': float(movie[4]) if movie[4] is not None else 0.0,
                        'vote_average': float(movie[5]) if movie[5] is not None else 0.0,
                        'release_date': movie[6],
                        'language': movie[7],
                        'runtime': movie[8],
                        'vote_count': movie[9]
                    }

                if cast is not None:
                    lookup_table[cast[0]]['cast'] = cast[1] if cast[1] is not None else []

                if prod is not None:
                    lookup_table[prod[0]]['prod'] = prod [1] if prod[1] is not None else []

                if genre is not None:
                    lookup_table[genre[0]]['genre'] = genre[1] if genre[1] is not None else []


            features: List[MovieFeatures] = []
            for movie_id in movie_ids:
                movie_data = lookup_table.get(movie_id, {})
                if movie_data:
                    features.append(MovieFeatures(
                        movie_id=movie_data.get('movie', {}).get('id'),
                        title=movie_data.get('movie', {}).get('title'),
                        overview=movie_data.get('movie', {}).get('overview'),
                        genre=movie_data.get('movie', {}).get('genre'),
                        popularity=movie_data.get('movie', {}).get('popularity'),
                        vote_average=movie_data.get('movie', {}).get('vote_average'),
                        release_date=movie_data.get('movie', {}).get('release_date'),
                        year=movie_data.get('movie', {}).get('release_date').year if movie_data.get('movie', {}).get('release_date') else None,
                        language=movie_data.get('movie', {}).get('language'),
                        runtime=movie_data.get('movie', {}).get('runtime'),
                        vote_count=movie_data.get('movie', {}).get('vote_count'),
                        cast_ids=movie_data.get('cast', []),
                        production_companies=movie_data.get('prod', []),
                        genres=movie_data.get('genre', [])
                    ))

            return features

        except Exception as e:
            print(e)
            logger.error(f"Error getting movie features: {str(e)}")
            raise CommonTasksError(f"Failed to get movie features: {str(e)}")
        
commonTasks = CommonTasks()