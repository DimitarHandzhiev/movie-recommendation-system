import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVDpp
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import  cosine_similarity
from  src.seed_movie_helper import SeedMovieHelper

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
        self.index_movie_map = {}

    def load_data(self):#loading data from dataset
        self.ratings_df = pd.read_csv(self.ratings_path)
        self.movies_df = pd.read_csv(self.movies_path)

        self.movies_df["genres_list"]= self.movies_df["genres"].fillna("").apply(lambda x: x.split("|") if x else [])
        self.movie_id_to_title = pd.Series(self.movies_df["title"].values, index=self.movies_df["movieId"]).to_dict()
        self.movie_id_to_genres = pd.Series(self.movies_df["genres"].values, index=self.movies_df["movieId"]).to_dict()
        self.movie_index_map = pd.Series(self.movies_df.index, index=self.movies_df["movieId"]).to_dict()
        self.index_movie_map = pd.Series(self.movies_df["movieId"].values, index=self.movies_df.index).to_dict()

        self.seed_helper = SeedMovieHelper(self.ratings_df, self.movies_df)

    def fit_content_model(self):#initializing content model
        self.genre_mlb = MultiLabelBinarizer()
        self.movie_genre_matrix = self.genre_mlb.fit_transform(self.movies_df["genres_list"])

    def fit_svdpp_model(self):#SVD++ structure model

        reader = Reader(rating_scale=(0.5,5.0))

        data = Dataset.load_from_df(self.ratings_df[["userId", "movieId", "rating"]], reader)

        trainset = data.build_full_trainset()
        self.svdpp_model = SVDpp(n_factors = 50, n_epochs = 20, lr_all = 0.005, reg_all = 0.02, random_state = 42)
        self.svdpp_model.fit(trainset)

    def fit(self):
        self.load_data()
        self.fit_content_model()
        self.fit_svdpp_model()

    def compute_new_user_vector(self, user_ratings: dict):
        #Build new user from given ratings

        if not user_ratings:
            return None

        item_vectors = []
        weights = []
        for movie_id, rating in user_ratings.items():
            try:
                inner_iid = self.svdpp_model.trainset.to_inner_iid(movie_id)
            except:
                continue
            qi = self.svdpp_model.qi[inner_iid]
            item_vectors.append(qi)
            weights.append(float(rating))

        if not item_vectors:
            return None

        item_vectors = np.array(item_vectors)
        weights = np.array(weights).reshape(-1,1)

        user_vector = np.sum(item_vectors * weights, axis=0)

        norm = np.linalg.norm(user_vector)
        if norm >0:
            user_vector = user_vector / norm

        return user_vector

    def compute_svdpp_scores_new_user(self, user_ratings: dict):
        #new user predictions
        user_vector = self.compute_new_user_vector(user_ratings)
        if user_vector is None:
            return pd.DataFrame(columns=["movieId", "svdpp_score"])

        rows = []
        for movie_id in self.movies_df["movieId"]:
            if movie_id in user_ratings:
                continue
            try:
                inner_iid = self.svdpp_model.trainset.to_inner_iid(movie_id)
            except:
                continue

            qi = self.svdpp_model.qi[inner_iid]
            bi = self.svdpp_model.bi[inner_iid]
            mu = self.svdpp_model.trainset.global_mean

            score = mu + bi + np.dot(user_vector, qi)

            rows.append({"movieId": movie_id, "svdpp_score": float(score)})

            return pd.DataFrame(rows)

    def build_user_content_profile(self, user_ratings: dict):
        #builds the content scores
        if not user_ratings:
            return None

        vectors = []
        weights = []

        for movie_id, rating in user_ratings.items():
            if movie_id not in self.movie_index_map:
                continue
            idx = self.movie_index_map[movie_id]
            vec = self.movie_genre_matrix[idx]

            vectors.append(vec)
            weights.append(float(rating))

        if not vectors:
            return None

        vectors = np.array(vectors)
        weights = np.array(weights).reshape(-1,1)

        profile = np.sum(vectors * weights, axis=0)

        norm = np.linalg.norm(profile)
        if norm>0:
            profile = profile /norm

        return profile

    def compute_content_scores(self, user_ratings: dict):
        profile = self.build_user_content_profile(user_ratings)

        if profile is None:
            return pd.DataFrame(columns=["movieId","content_score"])

        similarities = cosine_similarity([profile], self.movie_genre_matrix)[0]

        return pd.DataFrame({"movieId": self.movies_df["movieId"], "content_score": similarities})

    def recommend_from_ratings(self, user_ratings: dict, top_n: int = 10, alpha: float = 0.7):
        #final hybrid mode

        svdpp_scores = self.compute_svdpp_scores_new_user(user_ratings)
        content_scores = self.compute_content_scores(user_ratings)

        merged = pd.merge(svdpp_scores, content_scores, on="movieId", how="outer").fillna(0.0)

        merged = merged[~merged["movieId"].isin(user_ratings.keys())]

        for col in ["svdpp_score", "content_score"]:
            if merged[col].max() > merged[col].min():
                merged[col]= (merged[col] - merged[col].min()) / (merged[col].max() - merged[col].min())
            else:
                merged[col]= 0.0

        merged["final_score"] = alpha * merged["svdpp_score"] + (1 - alpha) * merged["content_score"]

        merged = merged.sort_values("final_score", ascending=False).head(top_n)

        merged = merged.merge(self.movies_df[["movieId", "title", "genres"]], on="movieId")

        return merged[["movieId", "title", "genres", "svdpp_score", "content_score", "final_score"]].reset_index(drop=True)

    def get_random_seed_movies(self, **kwargs):
        return self.seed_helper.get_random_seed_movies(**kwargs)

    def get_replacement_movie(self, **kwargs):
        return self.seed_helper.get_replacement_movie(**kwargs)