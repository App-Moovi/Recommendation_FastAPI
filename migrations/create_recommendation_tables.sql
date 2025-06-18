-- Add missing foreign key constraint
ALTER TABLE movie_preferences 
ADD CONSTRAINT fk_movie_preferences_movie 
FOREIGN KEY (movie_id) REFERENCES movies(id) ON DELETE CASCADE;

-- Fix user_languages primary key (should be composite)
ALTER TABLE user_languages DROP CONSTRAINT user_languages_pkey;
ALTER TABLE user_languages ADD PRIMARY KEY (language_code, user_id);

-- Create recommendation cache table
CREATE TABLE IF NOT EXISTS recommendation_cache (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    movie_id INTEGER NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    score DECIMAL(10, 6) NOT NULL,
    recommendation_reason TEXT,
    matched_user_ids INTEGER[], -- Users who influenced this recommendation
    position INTEGER NOT NULL, -- Position in recommendation list (1-40)
    batch_number INTEGER NOT NULL, -- Which batch (1 or 2)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    UNIQUE(user_id, movie_id, batch_number)
);

-- Create user similarity table for pre-computed similarities
CREATE TABLE IF NOT EXISTS user_similarities (
    user_id_1 INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    user_id_2 INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    similarity_score DECIMAL(5, 4) NOT NULL, -- 0.0000 to 1.0000
    common_movies INTEGER NOT NULL DEFAULT 0,
    last_calculated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id_1, user_id_2),
    CHECK (user_id_1 < user_id_2) -- Ensure we only store one direction
);

-- Create potential matches table
CREATE TABLE IF NOT EXISTS potential_matches (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    movie_id INTEGER NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    potential_match_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    match_condition VARCHAR(50) NOT NULL, -- 'like', 'superlike', 'rating_3+', 'rating_4+', 'dislike'
    confidence_score DECIMAL(5, 4) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    UNIQUE(user_id, movie_id, potential_match_user_id, match_condition)
);

-- Create movie similarities table for content-based filtering
CREATE TABLE IF NOT EXISTS movie_similarities (
    movie_id_1 INTEGER NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    movie_id_2 INTEGER NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    similarity_score DECIMAL(5, 4) NOT NULL,
    last_calculated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (movie_id_1, movie_id_2),
    CHECK (movie_id_1 < movie_id_2)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_recommendation_cache_user_expires 
ON recommendation_cache(user_id, expires_at);

CREATE INDEX IF NOT EXISTS idx_recommendation_cache_batch 
ON recommendation_cache(user_id, batch_number, position);

CREATE INDEX IF NOT EXISTS idx_user_similarities_score 
ON user_similarities(similarity_score DESC);

CREATE INDEX IF NOT EXISTS idx_user_similarities_user1 
ON user_similarities(user_id_1);

CREATE INDEX IF NOT EXISTS idx_user_similarities_user2 
ON user_similarities(user_id_2);

CREATE INDEX IF NOT EXISTS idx_potential_matches_user_movie 
ON potential_matches(user_id, movie_id);

CREATE INDEX IF NOT EXISTS idx_movie_similarities_movie1 
ON movie_similarities(movie_id_1);

CREATE INDEX IF NOT EXISTS idx_movie_similarities_movie2 
ON movie_similarities(movie_id_2);

-- Indexes on existing tables for better query performance
CREATE INDEX IF NOT EXISTS idx_movie_ratings_user 
ON movie_ratings(user_id);

CREATE INDEX IF NOT EXISTS idx_movie_preferences_user 
ON movie_preferences(user_id);

CREATE INDEX IF NOT EXISTS idx_movie_list_genres_genre 
ON movie_list_genres(genre_id);

CREATE INDEX IF NOT EXISTS idx_movies_release_date 
ON movies(release_date);

CREATE INDEX IF NOT EXISTS idx_movies_popularity 
ON movies(popularity DESC);

-- Materialized view for user interaction summary
CREATE MATERIALIZED VIEW IF NOT EXISTS user_interaction_summary AS
SELECT 
    u.id as user_id,
    COUNT(DISTINCT mp.movie_id) as total_preferences,
    COUNT(DISTINCT mr.movie_id) as total_ratings,
    AVG(mr.rating) as avg_rating,
    COUNT(DISTINCT CASE WHEN mp.preference = 'LIKE' THEN mp.movie_id END) as liked_movies,
    COUNT(DISTINCT CASE WHEN mp.preference = 'SUPERLIKE' THEN mp.movie_id END) as superliked_movies,
    COUNT(DISTINCT CASE WHEN mp.preference = 'DISLIKE' THEN mp.movie_id END) as disliked_movies,
    ARRAY_AGG(DISTINCT ug.genre_id) as preferred_genres
FROM users u
LEFT JOIN movie_preferences mp ON u.id = mp.user_id
LEFT JOIN movie_ratings mr ON u.id = mr.user_id
LEFT JOIN user_genres ug ON u.id = ug.user_id
GROUP BY u.id;

CREATE UNIQUE INDEX ON user_interaction_summary(user_id);

-- Function to get user similarity
CREATE OR REPLACE FUNCTION get_user_similarity(user1_id INTEGER, user2_id INTEGER)
RETURNS DECIMAL AS $$
BEGIN
    RETURN (
        SELECT similarity_score 
        FROM user_similarities 
        WHERE (user_id_1 = LEAST(user1_id, user2_id) 
           AND user_id_2 = GREATEST(user1_id, user2_id))
    );
END;

$$ LANGUAGE plpgsql;