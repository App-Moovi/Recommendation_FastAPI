# Movie Recommender API

A sophisticated movie recommendation system with user matching capabilities built using FastAPI, PostgreSQL, and a hybrid recommendation approach.

## Features

- **Hybrid Recommendation Engine**: Combines collaborative filtering, content-based filtering, and popularity-based recommendations
- **User Matching**: Automatically matches users with similar movie tastes
- **Real-time Updates**: User interactions immediately affect matching decisions
- **Smart Caching**: Efficient caching system that stores 40 recommendations and serves 20 at a time
- **Diversity Optimization**: Ensures recommendations have variety across genres, years, and languages
- **Background Jobs**: Pre-computes similarities for better performance
- **Scalable Architecture**: Designed to handle up to 1000 concurrent users

## Architecture

### Recommendation Algorithm

The system uses a hybrid approach with weighted scoring:

- **Collaborative Filtering (50%)**: Recommendations based on similar users' preferences
- **Content-Based Filtering (30%)**: Recommendations based on movie features (genre, cast, etc.)
- **Popularity Score (20%)**: Ensures quality recommendations for new users

### User Matching

Users are matched based on:

- Shared movie interactions (likes, dislikes, ratings)
- Similarity scores pre-computed nightly
- Real-time matching when users interact with recommended movies
- 80% similarity threshold for matching (configurable)

### Interaction Weights

- **Superlike**: +3.0 (equivalent to 5-star rating)
- **Like**: +2.0
- **5-star rating**: +3.0
- **4-star rating**: +2.0
- **3-star rating**: +1.0
- **2-star rating**: -1.0
- **1-star rating**: -2.0 (equivalent to dislike)
- **Dislike**: -2.0
- **Seen**: +0.5
- **Hide**: -3.0

## Setup

### Prerequisites

- Python 3.8+
- PostgreSQL 12+
- pip

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd movie_recommender
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your database credentials
```

5. Run database migrations:

```bash
psql -U your_username -d moovie -f migrations/create_recommendation_tables.sql
```

6. Start the application:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Get Recommendations

```http
POST /api/v1/recommendations/
Content-Type: application/json

{
    "user_id": 123,
    "force_refresh": false,
    "include_movie_details": true
}
```

Response includes:

- 20 movie recommendations
- Score and reason for each recommendation
- Matched users who influenced the recommendation
- Potential matches if user interacts with the movie

### Get Next Batch

```http
GET /api/v1/recommendations/{user_id}/next?batch_number=2
```

### Record Interaction

```http
POST /api/v1/recommendations/{user_id}/interact
Content-Type: application/json

{
    "movie_id": 456,
    "interaction_type": "LIKE",
    "rating": 4.5  // Optional
}
```

### Get User Matches

```http
GET /api/v1/recommendations/{user_id}/matches?limit=10
```

### Health Check

```http
GET /api/v1/health/
GET /api/v1/health/db
```

## Background Jobs

The system runs the following scheduled tasks:

1. **User Similarity Computation** (Daily at 00:00)

   - Computes pairwise similarities between all users
   - Updates the `user_similarities` table

2. **Movie Similarity Computation** (Daily at 01:00)

   - Computes content-based similarities between movies
   - Updates the `movie_similarities` table

3. **Materialized View Refresh** (Every 6 hours)

   - Refreshes `user_interaction_summary` view

4. **Cache Cleanup** (Hourly)
   - Removes expired recommendations and potential matches

## Database Schema Extensions

The system creates the following additional tables:

- `recommendation_cache`: Stores pre-computed recommendations
- `user_similarities`: Pre-computed user similarity scores
- `potential_matches`: Stores potential user matches for each recommendation
- `movie_similarities`: Pre-computed movie similarity scores
- `user_interaction_summary`: Materialized view for quick user profile access

## Configuration

Key configuration options in `.env`:

- `DATABASE_URL`: PostgreSQL connection string
- `RECOMMENDATIONS_PER_REQUEST`: Movies per API request (default: 20)
- `CACHE_SIZE`: Total cached recommendations (default: 40)
- `MATCH_THRESHOLD`: Minimum similarity for matching (default: 0.8)
- `ENABLE_BACKGROUND_JOBS`: Enable/disable scheduled tasks

## Performance Considerations

1. **Indexes**: The system creates multiple indexes for optimal query performance
2. **Caching**: Recommendations are cached until midnight or all 20 are interacted with
3. **Pre-computation**: Similarities are computed nightly to avoid real-time calculations
4. **Connection Pooling**: Database connections are pooled for better concurrency

## Monitoring

Monitor the following for optimal performance:

- API response times (target: <1s)
- Background job execution times
- Database query performance
- Cache hit rates

## Scaling Recommendations

For larger deployments:

1. Use Redis for caching instead of database tables
2. Implement distributed computing for similarity calculations
3. Use read replicas for database queries
4. Consider microservice architecture for recommendation engine

## API Documentation

Once running, visit:

- Swagger UI: `http://localhost:8000/api/v1/docs`
- ReDoc: `http://localhost:8000/api/v1/redoc`
