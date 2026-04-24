from src.hybrid_svd import AdvancedHybridRecommender

recommender = AdvancedHybridRecommender(
    ratings_path="../data/ml-latest-small/ratings.csv",
    movies_path="../data/ml-latest-small/movies.csv"
)

recommender.fit()

seed_movies = recommender.get_random_seed_movies(n=5, random_state=42)

print("Seed movies:")
print(seed_movies[["movieId", "title"]])

user_ratings = {
    int(seed_movies.iloc[0]["movieId"]): 5.0,
    int(seed_movies.iloc[1]["movieId"]): 4.5,
    int(seed_movies.iloc[2]["movieId"]): 3.5,
    int(seed_movies.iloc[3]["movieId"]): 4.0,
    int(seed_movies.iloc[4]["movieId"]): 2.5,
}

print("\nUser ratings:")
print(user_ratings)

recommendations = recommender.recommend_from_ratings(user_ratings, top_n=5)

print("\nRecommendations:")
print(recommendations)