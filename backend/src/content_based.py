import pandas as pd
import difflib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self, movies_path:str):
        self.movies_path = movies_path
        self.movies_df = None
        self.similarity_matrix = None
        self.title_to_index = {}
        self.index_to_title = {}
        self.lower_title_to_original = {}

    def load_data(self):
        self.movies_df = pd.read_csv(self.movies_path) #loads movies from dataset
        #genres for each movie in the dataset are separated with |, we need to separate them with spaces
        # replace | separator with spaces, so CountVectorizer can tokenize genres
        self.movies_df["genres_clean"] = self.movies_df["genres"].str.replace("|"," ", regex=False)
        self.movies_df["genres_clean"] = self.movies_df["genres_clean"].fillna("") #handles missing values(this dataset does not have missing values)

    def build_similarity_matrix(self):
        vectorizer = CountVectorizer(token_pattern=r"[^ ]+")#tokenize the genres
        genre_matrix = vectorizer.fit_transform(self.movies_df["genres_clean"])#creates genre matrix with vectorized genres
        self.similarity_matrix = cosine_similarity(genre_matrix,genre_matrix)# builds the similarity matrix, using cosine similarity

        self.title_to_index = pd.Series(self.movies_df.index, index=self.movies_df["title"]).to_dict()
        self.index_to_title = pd.Series(self.movies_df["title"], index=self.movies_df.index).to_dict()

        self.lower_title_to_original = { # prevents case-sensitivity
            title.lower(): title for title in self.movies_df["title"]
        }

    def fit(self): # full pipeline, first load data then builds similarity matrix
        self.load_data()
        self.build_similarity_matrix()

    def find_exact_or_case_insensitive_match(self,user_input: str):
        if user_input in self.title_to_index:
            return user_input # return the exact title if it exists

        lower_input = user_input.lower()

        if lower_input in self.lower_title_to_original:
            return self.lower_title_to_original[lower_input]# return case-insensitive match if found

        return None

    def find_close_matches(self, user_input:str, n_matches:int = 5):
        titles = self.movies_df["title"].tolist()

        partial_matches = [# return partial substring matches first
            title for title in titles if user_input.lower() in title.lower()
        ]
        if partial_matches:
            return partial_matches[:n_matches]

        # returns fuzzy mathes(misspelled titles or closest titles to the user_input based on distance measure) if not partial matches found
        return difflib.get_close_matches(user_input, titles, n=n_matches, cutoff=0.4)

    def recommend_by_title(self, title: str, top_n: int = 10):
        if self.movies_df is None or self.similarity_matrix is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        matched_title = self.find_exact_or_case_insensitive_match(title)

        if matched_title is None:
            suggestions = self.find_close_matches(title)

            return {
                "error": f"Movie '{title}' not found.", "suggestions": suggestions
            }

        movie_index = self.title_to_index[matched_title]

        similarity_scores = list(enumerate(self.similarity_matrix[movie_index])) # get similarity scores for the selected movie
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True) # Sort by similarity score descending
        similarity_scores = [x for x in similarity_scores if x[0] != movie_index] # remove the movie itself
        top_matches = similarity_scores[:top_n] #keep only top_n

        recommended_indices = [idx for idx, score in top_matches]
        recommended_scores = [score for idx, score in top_matches]

        recommendations = self.movies_df.loc[recommended_indices, ["movieId", "title", "genres"]].copy()
        recommendations["similarity_score"]= recommended_scores

        return recommendations.reset_index(drop=True)

    def get_all_titles(self):
        if self.movies_df is None:
            raise ValueError("Data not loaded. Call fit() first!")

        return sorted(self.movies_df["titles"].tolist())