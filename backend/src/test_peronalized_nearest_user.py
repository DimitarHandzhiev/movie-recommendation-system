from src.personalized_nearest_user import PersonalizedNearestUserRecommender

recommender = PersonalizedNearestUserRecommender(
    ratings_path="../data/ml-latest-small/ratings.csv",
    movies_path="../data/ml-latest-small/movies.csv"
)

recommender.fit()

seed_movies = recommender.get_valid_random_seed_movies(
    n=5,
    min_ratings=100,
    min_avg_rating=3.5,
    min_overlap_movies=2,
    min_candidate_users=20,
    max_attempts=30
)
print("Seed movies:")
print(seed_movies[["movieId", "title", "genres", "rating_count", "avg_rating"]])

user_ratings = {
    int(seed_movies.iloc[0]["movieId"]): 4.5,
    int(seed_movies.iloc[1]["movieId"]): 5.0,
    int(seed_movies.iloc[2]["movieId"]): 3.5,
    int(seed_movies.iloc[3]["movieId"]): 4.0,
    int(seed_movies.iloc[4]["movieId"]): 2.5,
}

print("\nUser ratings:")
print(user_ratings)

nearest_users = recommender.find_nearest_user(user_ratings, top_k=10)
print("\nNearest users:")
print(nearest_users)

recommendations = recommender.recommend_from_ratings(user_ratings, top_k_users=10, top_n=5)
print("\nRecommendations:")
print(recommendations[["title", "genres", "score"]])