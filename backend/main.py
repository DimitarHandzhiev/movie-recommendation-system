from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
import os
import requests

from src.content_based import ContentBasedRecommender
from src.personalized_nearest_user import PersonalizedNearestUserRecommender
from src.hybrid_svd import AdvancedHybridRecommender
from src.seed_movie_helper import SeedMovieHelper

import pandas as pd

RATINGS_PATH = "data/ml-latest-small/ratings.csv"
MOVIES_PATH = "data/ml-latest-small/movies.csv"
LINKS_PATH = "data/ml-latest-small/links.csv"
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

app = FastAPI()
print("Wait until models fit, then start frontend!")
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
links_df = pd.read_csv(LINKS_PATH)

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
    records = seed_movies.to_dict(orient="records")
    return add_movie_metadata(records)

class ReplaceMovieRequest(BaseModel):
    exclude_movie_ids: list[int] = []

@app.post("/replace-movie")
def replace_movie(req: ReplaceMovieRequest):
    movie = seed_helper.get_replacement_movie(exclude_movie_ids=req.exclude_movie_ids, min_ratings=100, min_avg_rating=3.5,)
    if movie is None:
        return {"error": "No replacement movie available"}

    return add_movie_metadata([movie])[0]
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

        records = recs.to_dict(orient="records")
        return add_movie_metadata(records)
    except Exception as e:
        return {"error": str(e)}

poster_cache = {}

def get_poster_url(movie_id: int):
    if movie_id in poster_cache:
        return poster_cache[movie_id]
    if not TMDB_API_KEY:
        poster_cache[movie_id] = None
        return None

    row = links_df[links_df["movieId"] ==movie_id]

    if row.empty or pd.isna(row.iloc[0]["tmdbId"]):
        poster_cache[movie_id] = None
        return None

    tmdb_id = int(row.iloc[0]["tmdbId"])

    try:
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
        response = requests.get(url, params={"api_key": TMDB_API_KEY}, timeout=5,)

        if response.status_code != 200:
            poster_cache[movie_id]=None
            return None

        data = response.json()
        poster_path = data.get("poster_path")
        if not poster_path:
            poster_cache[movie_id] = None
            return None

        poser_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
        poster_cache[movie_id] = poser_url
        return poser_url
    except Exception:
        poster_cache[movie_id]=None
        return None

def add_poster_urls(records):
    for record in records:
        record["posterUrl"] = get_poster_url(int(record["movieId"]))
    return records

movie_rating_stats = (ratings_df.groupby("movieId")["rating"].mean().reset_index().rename(columns={"rating":"avgRating"}))

def get_imdb_url(movie_id:int):
    row = links_df[links_df["movieId"]== movie_id]

    if row.empty or pd.isna(row.iloc[0]["imdbId"]):
        return None
    imdb_id = int(row.iloc[0]["imdbId"])
    return f"https://www.imdb.com/title/tt{imdb_id:07d}/"

def get_avg_rating(movie_id: int):
    row = movie_rating_stats[movie_rating_stats["movieId"]== movie_id]

    if row.empty:
        return None
    return round(float(row.iloc[0]["avgRating"]), 1)

def add_movie_metadata(records):
    for record in records:
        movie_id = int(record["movieId"])
        record["posterUrl"] = get_poster_url(movie_id)
        record["imdbUrl"] = get_imdb_url(movie_id)
        record["avgRating"] = get_avg_rating(movie_id)

    return records