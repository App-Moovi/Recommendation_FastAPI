from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, ARRAY, Text, DECIMAL, text, bindparam
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timezone
from typing import List, Tuple, Optional
from sqlalchemy.orm import Session
from app.database import SessionLocal

Base = declarative_base()

class RecommendationCache(Base):
    __tablename__ = "recommendation_cache"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    movie_id = Column(Integer, ForeignKey("movies.id", ondelete="CASCADE"), nullable=False)
    score = Column(DECIMAL(10, 6), nullable=False)
    recommendation_reason = Column(Text)
    matched_user_ids = Column(ARRAY(Integer))
    position = Column(Integer, nullable=False)
    batch_number = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)

class UserSimilarities(Base):
    __tablename__ = "user_similarities"
    
    user_id_1 = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    user_id_2 = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    similarity_score = Column(DECIMAL(5, 4), nullable=False)
    common_movies = Column(Integer, nullable=False, default=0)
    last_calculated = Column(DateTime, default=datetime.utcnow)

class PotentialMatches(Base):
    __tablename__ = "potential_matches"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    movie_id = Column(Integer, ForeignKey("movies.id", ondelete="CASCADE"), nullable=False)
    potential_match_user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    match_condition = Column(String(50), nullable=False)
    confidence_score = Column(DECIMAL(5, 4), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)

# Stub definition (no other columns needed)
class Movies(Base):
    __tablename__ = "movies"

    id = Column(Integer, primary_key=True)

class MovieSimilarities(Base):
    __tablename__ = "movie_similarities"
    
    movie_id_1 = Column(Integer, ForeignKey("movies.id", ondelete="CASCADE"), primary_key=True)
    movie_id_2 = Column(Integer, ForeignKey("movies.id", ondelete="CASCADE"), primary_key=True)
    similarity_score = Column(DECIMAL(5, 4), nullable=False)
    last_calculated = Column(DateTime, default=datetime.now(timezone.utc))

    def __repr__(self):
        return f"MovieSimilarities(movie_id_1={self.movie_id_1}, movie_id_2={self.movie_id_2}, similarity_score={self.similarity_score})"

    @staticmethod
    def get_movie_similarity(movie1_id: int, movie2_id: int, db: Optional[Session] = None) -> float:
        db = db or SessionLocal()
        try:
            score = db.query(MovieSimilarities).filter(MovieSimilarities.movie_id_1 == movie1_id, MovieSimilarities.movie_id_2 == movie2_id).one().similarity_score
            return float(score)
        except Exception as e:
            raise e
        
    @staticmethod
    def list_movie_similarities(combinations: List[Tuple[int, int]], db: Optional[Session] = None) -> List[Tuple[int, int, float]]:
        db = db or SessionLocal()

        print(combinations)
        if not combinations:
            return []

        try:
            similarity_query = text("""
                SELECT movie_id_1, movie_id_2, similarity_score
                FROM movie_similarities
                WHERE (movie_id_1, movie_id_2) IN :combinations
            """).bindparams(bindparam("combinations", expanding=True))

            similarities = db.execute(similarity_query, {"combinations": combinations}).fetchall()
            return list(map(lambda x: (int(x[0]), int(x[1]), float(x[2])), similarities))
        except Exception as e:
            raise e