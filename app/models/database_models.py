from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, ARRAY, Text, Boolean, DECIMAL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

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

class MovieSimilarities(Base):
    __tablename__ = "movie_similarities"
    
    movie_id_1 = Column(Integer, ForeignKey("movies.id", ondelete="CASCADE"), primary_key=True)
    movie_id_2 = Column(Integer, ForeignKey("movies.id", ondelete="CASCADE"), primary_key=True)
    similarity_score = Column(DECIMAL(5, 4), nullable=False)
    last_calculated = Column(DateTime, default=datetime.utcnow)