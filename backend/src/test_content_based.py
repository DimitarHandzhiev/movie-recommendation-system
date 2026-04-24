from content_based import ContentBasedRecommender
import pandas as pd

recommender = ContentBasedRecommender("../data/ml-latest-small/movies.csv")
recommender.fit()

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

results = recommender.recommend_by_title("back to the future (1985)", top_n=5)
print(results)