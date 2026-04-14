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
