# Movie Recommendation System

A hybrid movie recommendation system built with Python and Surprise library using the MovieLens dataset.

## Goal
The goal of this project is to build a recommendation engine that combines collaborative filtering and content-based.

## Dataset
- MovieLens ml-latest-small

## Tech stack
- Python
- pandas
- numpy
- surprise
- scikit-learn

## 3 modes of work
- Mode 1 - Content-Based Filtering: Uses genre similarity between movies(User types-in a movie, he had watched and the engine suggest 5 similar movies, based on similar genre)
- Mode 2 - Nearest-User Collaborative Filltering: Finds users with similar ratings and recommends based on their preferences(User has to rate 5 random, well-known films from 0 to 5 and the engine finds user who has rated similar films with similar rates, and generates suggestions based on the preferences of this nearest-user)
- Mode 3 - Advanced Hybrid(SVD++): Combines SVD++ collaborative filtering(implicit+explicit feedback), genre-based content similarity, and cold-start user embedding from rated items.(User rates 5 randomly generated, well-known movies from 0 to 5, engine calculates final score taking into consideration implicit and explicit feedbacks, as well as content-based filtering)

## Evaluation:
Evaluation shows, that increasing the number of candidate items initially improves performance, but beyond a certain threshold introduces noise and degrades recommendation quality.
The best performance was achieved with approximately 5000 candidate items, highlighting the importance of effective candidate selection in recommendation systems.

### Evaluation results for 5000 candidates:
 Mode          | hitRate@5 | HitRate@10 | Precision@10
#------------------------------------------------------
 Content-based | 0.040     | 0.060      | 0.006
 Nearest-user  | 0.320     | 0.400      | 0.044
 Hybrid        | 0.160     | 0.300      | 0.032