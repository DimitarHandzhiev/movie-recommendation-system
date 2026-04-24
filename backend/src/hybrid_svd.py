import pandas as pd
import numpy as np

from surprise import Dataset, Reader, SVDpp
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

from src.seed_movie_helper import SeedMovieHelper


class AdvancedHybridRecommender:
    def __init__(self, ratings_path: str, movies_path: str):
        self.ratings_path = ratings_path
        self.movies_path = movies_path
        self.ratings_df = None
        self.movies_df = None
        self.movie_id_to_title = {}
        self.movie_id_to_genres = {}
        self.seed_helper = None
        self.svdpp_model = None
        self.genre_mlb = None
        self.movie_genre_matrix = None
        self.movie_index_map = {}

    def load_data(self):#loads data
        self.ratings_df = pd.read_csv(self.ratings_path)
        self.movies_df = pd.read_csv(self.movies_path)
        self.movies_df["genres_list"] = self.movies_df["genres"].fillna("").apply(lambda x: x.split("|") if x else [])
        self.movie_id_to_title = pd.Series(self.movies_df["title"].values, index=self.movies_df["movieId"]).to_dict()
        self.movie_id_to_genres = pd.Series(self.movies_df["genres"].values, index=self.movies_df["movieId"]).to_dict()
        self.movie_index_map = pd.Series(self.movies_df.index,index=self.movies_df["movieId"]).to_dict()
        self.seed_helper = SeedMovieHelper(self.ratings_df, self.movies_df)


    def fit_content_model(self):#content-based model
        self.genre_mlb = MultiLabelBinarizer()
        self.movie_genre_matrix = self.genre_mlb.fit_transform(self.movies_df["genres_list"])


    def fit_svdpp_model(self):#SVD++ model
        reader = Reader(rating_scale=(0.5, 5.0))

        data = Dataset.load_from_df(self.ratings_df[["userId", "movieId", "rating"]],reader)

        trainset = data.build_full_trainset()

        self.svdpp_model = SVDpp(
            n_factors=50,
            n_epochs=20,
            lr_all=0.005,
            reg_all=0.02,
            random_state=42
        )

        self.svdpp_model.fit(trainset)

    def fit(self):
        self.load_data()
        self.fit_content_model()
        self.fit_svdpp_model()


    def compute_user_similarity(self, target_ratings: dict, candidate_user_ratings: pd.DataFrame):#find nearest user
        score = 0.0
        overlap = 0

        for _, row in candidate_user_ratings.iterrows():
            movie_id = int(row["movieId"])

            if movie_id in target_ratings:
                diff = abs(target_ratings[movie_id] - float(row["rating"]))
                contribution = max(0.0, 5.0 - diff)
                score += contribution
                overlap += 1

        if overlap < 2:
            return 0.0

        return score * overlap

    def find_nearest_users(self, user_ratings: dict, top_k: int = 10):
        candidate_users = self.ratings_df[self.ratings_df["movieId"].isin(user_ratings.keys())]["userId"].unique()

        similarities = []

        for user_id in candidate_users:
            user_rows = self.ratings_df[self.ratings_df["userId"] == user_id]

            sim = self.compute_user_similarity(user_ratings, user_rows)

            if sim > 0:
                similarities.append((user_id, sim))

        if not similarities:
            return []

        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

        return similarities


    def compute_svdpp_scores(self, user_ratings: dict):#SVD++ score
        nearest_users = self.find_nearest_users(user_ratings)

        if not nearest_users:
            return pd.DataFrame(columns=["movieId", "svdpp_score"])

        # Map user -> similarity
        sim_map = {u: s for u, s in nearest_users}


        #Candidate movies = movies rated by nearest users
        candidate_df = self.ratings_df[
            self.ratings_df["userId"].isin(sim_map.keys())
        ]
        #more precise

        # Count how many times each movie appears among similar users
        movie_counts = (
            candidate_df.groupby("movieId")
            .size()
            .sort_values(ascending=False)
        )

        # Take top N most frequent movies
        candidate_movie_ids = movie_counts.head(3000).index.tolist()

        # Remove already rated movies
        candidate_movie_ids = [
            m for m in candidate_movie_ids if m not in user_ratings
        ]

        # Compute scores
        rows = []

        for movie_id in candidate_movie_ids:
            weighted_sum = 0.0
            sim_sum = 0.0

            for user_id, sim in sim_map.items():
                pred = self.svdpp_model.predict(uid=user_id, iid=movie_id).est

                weighted_sum += pred * sim
                sim_sum += sim

            if sim_sum > 0:
                score = weighted_sum / sim_sum
            else:
                score = 0.0

            rows.append({
                "movieId": movie_id,
                "svdpp_score": score
            })

        return pd.DataFrame(rows)

    # Content score
    def build_user_content_profile(self, user_ratings: dict):
        vectors = []
        weights = []

        for movie_id, rating in user_ratings.items():
            if movie_id not in self.movie_index_map:
                continue

            idx = self.movie_index_map[movie_id]
            vec = self.movie_genre_matrix[idx]
            vectors.append(vec)
            weights.append(rating)

        if not vectors:
            return None

        vectors = np.array(vectors)
        weights = np.array(weights).reshape(-1, 1)

        profile = np.sum(vectors * weights, axis=0)

        norm = np.linalg.norm(profile)
        if norm > 0:
            profile = profile / norm

        return profile

    def compute_content_scores(self, user_ratings: dict):
        profile = self.build_user_content_profile(user_ratings)

        if profile is None:
            return pd.DataFrame(columns=["movieId", "content_score"])

        similarities = cosine_similarity([profile],self.movie_genre_matrix)[0]

        return pd.DataFrame({
            "movieId": self.movies_df["movieId"],
            "content_score": similarities
        })


    def recommend_from_ratings(self, user_ratings: dict, top_n: int = 10, alpha: float = 0.95):#hybrid mode

        svdpp_scores = self.compute_svdpp_scores(user_ratings)
        content_scores = self.compute_content_scores(user_ratings)

        merged = pd.merge(
            svdpp_scores,
            content_scores,
            on="movieId",
            how="outer"
        ).fillna(0.0)

        merged = merged[~merged["movieId"].isin(user_ratings.keys())]

        # Normalize
        for col in ["svdpp_score", "content_score"]:
            if merged[col].max() > merged[col].min():
                merged[col] = (merged[col] - merged[col].min()) / (merged[col].max() - merged[col].min())
            else:
                merged[col] = 0.0

        merged["final_score"] = alpha * merged["svdpp_score"] + (1 - alpha) * merged["content_score"]

        merged = merged.sort_values("final_score", ascending=False).head(top_n)

        merged = merged.merge(
            self.movies_df[["movieId", "title", "genres"]],
            on="movieId"
        )

        return merged[[
            "movieId",
            "title",
            "genres",
            "svdpp_score",
            "content_score",
            "final_score"
        ]].reset_index(drop=True)

#seed movies
    def get_random_seed_movies(self, **kwargs):
        return self.seed_helper.get_random_seed_movies(**kwargs)

    def get_replacement_movie(self, **kwargs):
        return self.seed_helper.get_replacement_movie(**kwargs)