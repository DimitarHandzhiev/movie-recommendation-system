import pandas as pd
class PersonalizedNearestUserRecommender:
    def __init__(self, ratings_path: str, movies_path: str):
        self.ratings_path = ratings_path
        self.movies_path = movies_path
        self.ratings_df = None
        self.movies_df = None
        self.title_to_movie_id = {}
        self.movie_id_to_title = {}
        self.movie_id_to_genres = {}

    def load_data(self):
        self.ratings_df = pd.read_csv(self.ratings_path)
        self.movies_df = pd.read_csv(self.movies_path)

        self.title_to_movie_id = pd.Series(self.movies_df["movieId"].values, index=self.movies_df["title"]).to_dict()
        self.movie_id_to_title = pd.Series(self.movies_df["title"].values, index=self.movies_df["movieId"]).to_dict()
        self.movie_id_to_genres = pd.Series(self.movies_df["genres"].values, index=self.movies_df["movieId"]).to_dict()

        self.lower_title_to_original = {
            title.lower(): title for title in self.movies_df["title"]
        }

    def fit(self):#load the data
        self.load_data()

    def compute_user_similarity(self, target_ratings:dict, candidate_user_ratings: pd.DataFrame):
        #compute similarity between the input ratings and the ratings of one existing user
        score = 0.0
        overlap_count = 0

        for _, row in candidate_user_ratings.iterrows():
            movie_id = row["movieId"]
            existing_rating = float(row["rating"])

            if movie_id in target_ratings:
                target_rating = float(target_ratings[movie_id])
                diff = abs(target_rating - existing_rating)# smaller rating difference means higher similarity
                contribution = max(0.0, 5.0 - diff) # max usefull diff is roughly 4.5 on the 0.5-5.0 scale

                score += contribution
                overlap_count += 1

        if overlap_count < 2:
            return 0.0

        return score * overlap_count# reward stronger overplap

    def find_nearest_user(self, user_ratings: dict, top_k: int = 20):
        # find the users with most similar ratings as the inputed ratings

        if not user_ratings:
            return pd.DataFrame(columns=["userId", "similarity"])

        candidate_users = self.ratings_df[
            self.ratings_df["movieId"].isin(user_ratings.keys())]["userId"].unique()

        similarities = []

        for user_id in candidate_users:
            candidate_user_ratings = self.ratings_df[
                self.ratings_df["userId"] == user_id
            ]
            similarity = self.compute_user_similarity(target_ratings=user_ratings,candidate_user_ratings=candidate_user_ratings)
            if similarity > 0:
                similarities.append({
                    "userId":user_id,
                    "similarity": similarity
                })

        similarities_df = pd.DataFrame(similarities)

        if similarities_df.empty:
            return pd.DataFrame(columns=["userId", "similarity"])

        similarities_df = similarities_df.sort_values(
            by="similarity",
            ascending=False
        ).head(top_k)

        return similarities_df.reset_index(drop=True)

    def recommend_from_ratings(self, user_ratings: dict, top_k_users: int =20, top_n: int = 10):
        # generate recommendations from the ratings of a new user
        if not user_ratings:
            return pd.DataFrame(columns=["movieId", "title", "genres", "score"])


        nearest_users_df = self.find_nearest_user(user_ratings, top_k=top_k_users)

        if nearest_users_df.empty:
            return pd.DataFrame(columns=["movieId", "title", "genres", "score"])

        nearest_user_ids = nearest_users_df["userId"].tolist()
        similarity_map = pd.Series(nearest_users_df["similarity"].values, index=nearest_users_df["userId"]).to_dict()

        candidate_ratings = self.ratings_df[self.ratings_df["userId"].isin(nearest_user_ids)].copy()

        #remove movies already rated by the input user
        candidate_ratings = candidate_ratings[~candidate_ratings["movieId"].isin(user_ratings.keys())]

        if candidate_ratings.empty:
            return pd.DataFrame(columns=["movieId", "title", "genres", "score"])

        #weighted score = existing user's rating * that user's similarity
        candidate_ratings["weighted_score"] = candidate_ratings.apply(lambda row: float(row["rating"]) * similarity_map[row["userId"]], axis=1)

        grouped = candidate_ratings.groupby("movieId").agg(weighted_score_sum=("weighted_score", "sum"), rating_count=("rating", "count")).reset_index()

        # Reward movies supported by multiple similar users
        grouped["final_score"] = grouped["weighted_score_sum"] * grouped["rating_count"]

        grouped = grouped.sort_values(by="final_score", ascending=False).head(top_n)

        recommendations = grouped.merge(self.movies_df[["movieId", "title", "genres"]], on="movieId", how="left")

        recommendations = recommendations[["movieId", "title", "genres", "final_score"]].rename(columns={"final_score":"score"})

        return recommendations.reset_index(drop=True)

    def get_random_seed_movies(self, n: int = 5, min_ratings:int = 200, random_state=None, exclude_movie_ids=None):
        # return a random sample of popular and recognizable movies for the rating step
        if exclude_movie_ids is None:
            exclude_movie_ids = set()
        else:
            exclude_movie_ids = set(exclude_movie_ids)

        movie_stats = self.ratings_df.groupby("movieId").agg(rating_count=("rating", "count"), avg_rating=("rating","mean")).reset_index()

        # select only popular movies
        popular_movies = movie_stats[movie_stats["rating_count"]>=min_ratings].copy()
        popular_movies = popular_movies.merge(self.movies_df[["movieId", "title", "genres"]], on="movieId", how="left")
        # select only generally liked movies
        popular_movies = popular_movies[~popular_movies["movieId"].isin(exclude_movie_ids)]

        if popular_movies.empty:
            return pd.DataFrame(columns=["movieId", "title", "genres", "rating_count", "avg_rating"])

        if len(popular_movies) <= n:
            return popular_movies[["movieId", "title", "genres", "rating_count", "avg_rating"]].reset_index(drop=True)

        sampled = popular_movies.sample(n=n, random_state=random_state)

        return sampled[["movieId", "title", "genres", "rating_count", "avg_rating"]].reset_index(drop=True)

    def get_replacement_movie(self, exclude_movie_ids=None, min_ratings: int = 100):
        # return a single replacement movie in case of "Have not watched this movie!"
        replacement_df = self.get_random_seed_movies(n=1, min_ratings=min_ratings, exclude_movie_ids=exclude_movie_ids)

        if replacement_df.empty:
            return None

        return replacement_df.iloc[0].to_dict()

    def get_movie_details_by_ids(self, movie_ids: list):
        # return movie details for a list of movie IDs
        return self.movies_df[self.movies_df["movieId"].isin(movie_ids)][["movieId", "title", "genres"]].copy()

    def has_enough_overlap_users(self, seed_movie_ids:list, min_overlap_movies:int=2, min_candidate_users: int=20):
        if not seed_movie_ids:
            return False

        candidate_ratings = self.ratings_df[
            self.ratings_df["movieId"].isin(seed_movie_ids)
        ]

        overlap_counts = candidate_ratings.groupby("userId")["movieId"].nunique()

        strong_candidates = overlap_counts[overlap_counts >= min_overlap_movies]

        return len(strong_candidates) >= min_candidate_users

    def get_valid_random_seed_movies(
            self,
            n: int = 5,
            min_ratings: int = 100,
            min_avg_rating: float = 3.5,
            min_overlap_movies: int = 2,
            min_candidate_users: int = 20,
            max_attempts: int = 30,
            exclude_movie_ids=None,
            random_state=None
    ):
        if exclude_movie_ids is None:
            exclude_movie_ids = set()
        else:
            exclude_movie_ids = set(exclude_movie_ids)

        movie_stats = (
            self.ratings_df.groupby("movieId")["rating"]
            .agg(["count", "mean"])
            .reset_index()
            .rename(columns={
                "count": "rating_count",
                "mean": "avg_rating"
            })
        )

        popular_movies = movie_stats[movie_stats["rating_count"] >= min_ratings].copy()
        popular_movies = popular_movies[popular_movies["avg_rating"] >= min_avg_rating]

        popular_movies = popular_movies.merge(
            self.movies_df[["movieId", "title", "genres"]],
            on="movieId",
            how="left"
        )

        popular_movies = popular_movies[
            ~popular_movies["movieId"].isin(exclude_movie_ids)
        ].reset_index(drop=True)

        if popular_movies.empty:
            return pd.DataFrame(
                columns=["movieId", "title", "genres", "rating_count", "avg_rating"]
            )

        if len(popular_movies) < n:
            return popular_movies[
                ["movieId", "title", "genres", "rating_count", "avg_rating"]
            ].reset_index(drop=True)

        for attempt in range(max_attempts):
            # Use different randomness across attempts unless a fixed seed is explicitly desired
            current_random_state = None if random_state is None else random_state + attempt

            sampled = popular_movies.sample(n=n, random_state=current_random_state).reset_index(drop=True)
            seed_movie_ids = sampled["movieId"].tolist()

            if self.has_enough_overlap_users(
                    seed_movie_ids=seed_movie_ids,
                    min_overlap_movies=min_overlap_movies,
                    min_candidate_users=min_candidate_users
            ):
                return sampled[
                    ["movieId", "title", "genres", "rating_count", "avg_rating"]
                ].reset_index(drop=True)

        # Fallback: if no valid set is found, return one random sample anyway
        fallback = popular_movies.sample(n=n, random_state=random_state).reset_index(drop=True)

        return fallback[
            ["movieId", "title", "genres", "rating_count", "avg_rating"]
        ].reset_index(drop=True)