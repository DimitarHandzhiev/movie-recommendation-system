import pandas as pd


class SeedMovieHelper:
    def __init__(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        self.ratings_df = ratings_df
        self.movies_df = movies_df

    def _build_popular_movie_pool(self, min_ratings: int = 100, min_avg_rating: float = 3.5, exclude_movie_ids=None):
        #build a pool of popular and generally liked movies.
        if exclude_movie_ids is None:
            exclude_movie_ids = set()
        else:
            exclude_movie_ids = set(exclude_movie_ids)

        movie_stats = (self.ratings_df.groupby("movieId")["rating"].agg(["count", "mean"]).reset_index().rename(columns={"count": "rating_count", "mean": "avg_rating"}))
        pool = movie_stats[movie_stats["rating_count"] >= min_ratings].copy()
        pool = pool[pool["avg_rating"] >= min_avg_rating]

        pool = pool.merge(self.movies_df[["movieId", "title", "genres"]],on="movieId", how="left")
        pool = pool[~pool["movieId"].isin(exclude_movie_ids)].reset_index(drop=True)

        return pool

    def get_random_seed_movies(self,n: int = 5, min_ratings: int = 100, min_avg_rating: float = 3.5, exclude_movie_ids=None,random_state=None):
        #return a random sample of popular and recognizable movies.
        pool = self._build_popular_movie_pool(min_ratings=min_ratings,min_avg_rating=min_avg_rating,exclude_movie_ids=exclude_movie_ids)
        if pool.empty:
            return pd.DataFrame(
                columns=["movieId", "title", "genres", "rating_count", "avg_rating"])

        if len(pool) <= n:
            return pool[
                ["movieId", "title", "genres", "rating_count", "avg_rating"]].reset_index(drop=True)

        sampled = pool.sample(n=n, random_state=random_state)

        return sampled[
            ["movieId", "title", "genres", "rating_count", "avg_rating"]].reset_index(drop=True)

    def get_replacement_movie(self,exclude_movie_ids=None, min_ratings: int = 100, min_avg_rating: float = 3.5, random_state=None):
        # return one replacement movie for the 'I haven't watched this movie' action.
        replacement_df = self.get_random_seed_movies(n=1, min_ratings=min_ratings, min_avg_rating=min_avg_rating, exclude_movie_ids=exclude_movie_ids, random_state=random_state)
        if replacement_df.empty:
            return None

        return replacement_df.iloc[0].to_dict()