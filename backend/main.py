from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random

from src.content_based import ContentBasedRecommender
from src.personalized_nearest_user import PersonalizedNearestUserRecommender
from src.hybrid_svd import AdvancedHybridRecommender
from src.seed_movie_helper import SeedMovieHelper

import pandas as pd

RATINGS_PATH = "data/ml-latest-small/ratings.csv"
MOVIES_PATH = "data/ml-latest-small/movies.csv"

app = FastAPI()
print("LOADED MAIN.PY FROM BACKEND")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ratings_df = pd.read_csv(RATINGS_PATH)
movies_df = pd.read_csv(MOVIES_PATH)
seed_helper = SeedMovieHelper(ratings_df, movies_df)

content_model = ContentBasedRecommender(MOVIES_PATH)
content_model.fit()

nearest_model = PersonalizedNearestUserRecommender(RATINGS_PATH, MOVIES_PATH)
nearest_model.fit()
hybrid_model = AdvancedHybridRecommender(RATINGS_PATH, MOVIES_PATH)
hybrid_model.fit()
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/seed-movies")
def get_seed_movies():
    seed_movies = seed_helper.get_random_seed_movies(n=5, min_ratings=100, min_avg_rating = 3.5,)
    return seed_movies.to_dict(orient="records")

class ReplaceMovieRequest(BaseModel):
    exclude_movie_ids: list[int] = []

@app.post("/replace-movie")
def replace_movie(req: ReplaceMovieRequest):
    movie = seed_helper.get_replacement_movie(exclude_movie_ids=req.exclude_movie_ids, min_ratings=100, min_avg_rating=3.5,)
    if movie is None:
        return {"error": "No replacement movie available"}

    return movie
class RecommendRequest(BaseModel):
    mode: str
    ratings: dict
    top_n: int =10

@app.post("/recommend")
def recommend(req: RecommendRequest):
    user_ratings = {int(movie_id): float(rating) for movie_id, rating in req.ratings.items()}
    if not user_ratings:
        return {"error": "No ratings provided"}

    try:
        if req.mode=="nearest":
            recs = nearest_model.recommend_from_ratings(user_ratings=user_ratings,top_n=req.top_n,)
        elif req.mode == "hybrid":
            recs = hybrid_model.recommend_from_ratings(user_ratings=user_ratings, top_n=req.top_n,)
        elif req.mode =="content":
            first_movie_id = list(user_ratings.keys())[0]
            movie_title = movies_df.loc[movies_df["movieId"]==first_movie_id, "title"].values[0]
            recs = content_model.recommend_by_title(movie_title, top_n=req.top_n,)
        else:
            return {"error": "Invalid mode"}

        return recs.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}