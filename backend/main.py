from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random

from src.content_based import ContentBasedRecommender
from src.personalized_nearest_user import PersonalizedNearestUserRecommender
from src.hybrid_svd import AdvancedHybridRecommender

RATINGS_PATH = "data/ml-latest-small/ratings.csv"
MOVIES_PATH = "data/ml-latest-small/movies.csv"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/seed-movies")
def get_seed_movies():
    return [
        {"movieId": 590, "title": "Terminator 2"},
        {"movieId": 480, "title": "Jurassic Park"},
        {"movieId": 110, "title": "Braveheart"},
        {"movieId": 2858, "title": "American Beauty"},
        {"movieId": 5952, "title": "Lord of the Rings"}
    ]

class ReplaceMovieRequest(BaseModel):
    exclude_movie_ids: list[int] = []

@app.post("/replace-movie")
def replace_movie(req: ReplaceMovieRequest):
    fallback_movies=[
        {"movieId": 1, "title": "Toy Story (1995)", "genres": "Adventure|Animation|Children|Comedy|Fantasy"},
        {"movieId": 2, "title": "Jumanji (1995)", "genres": "Adventure|Children|Fantasy"},
        {"movieId": 3, "title": "Grumpier Old Men (1995)", "genres": "Comedy|Romance"},
        {"movieId": 5, "title": "Father of the Bride Part II (1995)", "genres": "Comedy"},
        {"movieId": 7, "title": "Sabrina (1995)", "genres": "Comedy|Romance"},
        {"movieId": 10, "title": "GoldenEye (1995)", "genres": "Action|Adventure|Thriller"},
        {"movieId": 32, "title": "Twelve Monkeys (1995)", "genres": "Mystery|Sci-Fi|Thriller"},
    ]
    candidates = [m for m in fallback_movies if m["movieId"] not in req.exclude_movie_ids]

    if not candidates:
        return {"error": "No replacement movie available"}

    return random.choice(candidates)