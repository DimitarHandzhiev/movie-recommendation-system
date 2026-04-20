import pandas as pd
import numpy as np
import random

from src.content_based import  ContentBasedRecommender
from src.personalized_nearest_user import PersonalizedNearestUserRecommender
from src.hybrid_svd import AdvancedHybridRecommender

RATINGS_PATH = "../data/ml-latest-small/ratings.csv"
MOVIES_PATH = "../data/ml-latest-small/movies.csv"
MIN_USER_RATINGS = 15
MIN_HIGH_RATINGS = 6
HIGH_RATING_THRESHOLD = 4.0
N_USERS = 50
TOP_K_LIST = [5,10]

def hit_rate(recommendations, ground_truth,k):
    rec_ids = set(recommendations[:k])
    return int(len(rec_ids.intersection(ground_truth))>0)

def precision_at_k(recommendations, ground_truth,k):
    rec_ids = set(recommendations[:k])
    return len(rec_ids.intersection(ground_truth)) / k

def prepare_users(ratings_df):
    user_counts = ratings_df.groupby("userId").size()
    valid_users = user_counts[user_counts >= MIN_USER_RATINGS].index.tolist()

    return valid_users

def sample_user_data(ratings_df, user_id):
    user_data = ratings_df[ratings_df["userId"] == user_id]

    high_rated = user_data[user_data["rating"] >= HIGH_RATING_THRESHOLD]

    if len(high_rated) < MIN_HIGH_RATINGS:
        return None

    movies = high_rated.sample(frac=1, random_state =42)

    seed = movies.iloc[:5]
    hidden = movies.iloc[5:8]
    seed_ratings = {int(row["movieId"]):float(row["rating"]) for _, row in seed.iterrows()}
    hidden_movies = set(hidden["movieId"].astype(int).tolist())

    return seed_ratings, hidden_movies

def evaluate():
    print("Loading data...")
    ratings_df = pd.read_csv(RATINGS_PATH)

    print("Preparing recommenders...")

    content_model = ContentBasedRecommender(MOVIES_PATH)
    content_model.fit()

    nearest_model = PersonalizedNearestUserRecommender(RATINGS_PATH, MOVIES_PATH)
    nearest_model.fit()

    hybrid_model = AdvancedHybridRecommender(RATINGS_PATH, MOVIES_PATH)
    hybrid_model.fit()

    valid_users = prepare_users(ratings_df)

    print(f"Total valid users: {len(valid_users)}")

    sampled_users = random.sample(valid_users, min(N_USERS, len(valid_users)))

    results = {"content": {"hr5": [], "hr10": [], "p10": []},
               "nearest": {"hr5":[], "hr10": [], "p10": []},
               "hybrid": {"hr5":[], "hr10": [], "p10": []},}

    print("Starting evaluation...")

    for i, user_id in enumerate(sampled_users):
        print(f"\n➡️ Starting user {i + 1}/{len(sampled_users)} (userId={user_id})")
        user_sample = sample_user_data(ratings_df, user_id)
        if user_sample is None:
            continue

        seed_ratings, hidden_movies = user_sample
        seed_movie_ids = list(seed_ratings.keys())

        #Content-based mode
        #--------------------
        seed_movie_id = seed_movie_ids[0]
        seed_movie_title = nearest_model.movie_id_to_title[seed_movie_id]
        content_recs = content_model.recommend_by_title(seed_movie_title, top_n=10)
        content_rec_ids = content_recs["movieId"].tolist()
        print("   Content done")

        #Nearest user mode
        #----------------------
        nearest_recs = nearest_model.recommend_from_ratings(seed_ratings, top_n=10)
        nearest_rec_ids = nearest_recs["movieId"].tolist()
        print("   Nearest done")

        #Hybrid mode
        #--------------
        hybrid_recs = hybrid_model.recommend_from_ratings(seed_ratings, top_n=10)
        hybrid_rec_ids = hybrid_recs["movieId"].tolist()
        print("   Hybrid done")

        for name, recs in zip(["content", "nearest", "hybrid"], [content_rec_ids, nearest_rec_ids, hybrid_rec_ids]):
            results[name]["hr5"].append(hit_rate(recs, hidden_movies, 5))
            results[name]["hr10"].append(hit_rate(recs, hidden_movies, 10))
            results[name]["p10"].append(precision_at_k(recs, hidden_movies, 10))

        print(f"Processed {i+1}/{len(sampled_users)} users", end="\r")

    print("\nEvaluation complete.\n")

    for model_name in results:
        hr5 = np.mean(results[model_name]["hr5"])
        hr10 = np.mean(results[model_name]["hr10"])
        p10 = np.mean(results[model_name]["p10"])

        print(f"=== {model_name.upper()} ===")
        print(f"HitRate@5:  {hr5:.3f}")
        print(f"HitRate@10: {hr10:.3f}")
        print(f"Precision@10: {p10:.3f}")
        print()

if __name__ == "__main__":
    evaluate()