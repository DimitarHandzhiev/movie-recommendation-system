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

##3 modes of work
- Mode 1 - Content-Based Filtering: Uses genre similarity between movies(User types-in a movie, he had watched and the engine suggest 5 similar movies, based on similar genre)
- Mode 2 - Nearest-User Collaborative Filltering: Finds users with similar ratings and recommends based on their preferences(User has to rate 5 random, well-known films from 0 to 5 and the engine finds user who has rated similar films with similar rates, and generates suggestions based on the preferences of this nearest-user)
- Mode 3 - Advanced Hybrid(SVD++): Combines SVD++ collaborative filtering(implicit+explicit feedback), genre-based content similarity, and cold-start user embedding from rated items.(User rates 5 randomly generated, well-known movies from 0 to 5, engine calculates final score taking into consideration implicit and explicit feedbacks, as well as content-based filtering)
