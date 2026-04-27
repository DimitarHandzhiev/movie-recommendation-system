# Movie Recommendation System

A full-stack movie recommendation system built with Python+FastAPI(backend) and React(frontend).
The system implements multiple recommendation algorithms and interactive UI.


## Features
There are 3 recommendation modes:
- Content-based Filtering
- Nearest-User (Collaborative Filtering)
- Hybrid Recommender (Combines content-based and collaborative filtering)

Interactive rating system:
- User can rate famous, well-known movies and based on its ratings, the system recommends 10 movies.

Modern UI:
- Blur animations
- Responsive movie cards

Movie posters(TMDB API):
- TMDB API key is used for movie poster pictures: Every movie card has as background the corresponding movie poster.
Since the sharing of personal API key is forbidden, the movie cards will appear without poster, with only black background.
Receiving a TMDB API key is easy with free registration on their website(https://www.themoviedb.org/).

IMDB Integration:
- By clicking on recommended movies you will be redirected to the corresponding movie page in IMDb.

## How it works:
1. User selects a recommendation mode
2. User rates 5 famous well-known movies. If the user has not watched the movie, there is a "Not Watched" option, which will replace the movie.
   (When using Content-based Recommender it is necessary to rate only the first movie) 
3. Selected recommender algorithm processes input
4. System returns personalized recommendations. 

Algorithms:
- Content-Based:
   - Based on genres similarity
- Nearest User:
   - Finds users with similar taste and recommends other movies that these users have watched
- Hybrid:
   - Combines both approached for better accuracy

## Dataset
- MovieLens dataset ml-latest-small
- TMDB API(for poster pictures)

## Tech stack
- Python
- pandas
- numpy
- surprise
- scikit-learn

## Setup & Run:
After starting the front and backend, you should wait until the models are fit before using the system. This could take 2-3minutes.
Keep in mind that different recommendation modes can take a couple seconds to give you recommendation movies. Be patient :)
- Just run start_app.bat for automatic start(this will start the back- and frontend and open browser automatically)

For manual setup:
- Backend:

   - cd backend
   - python -m venv .venv
   - .venv\Scripts\activate
   - pip install -r requirements.txt
   - uvicorn main:app --reload

- Frontend:

   - cd frontend
   - npm install
   - npm start

If you have TMDB API key and wants to visualize the movie posters:
- Uncomment the "TMDB_API_KEY=your_api_key_here" in the start_app.bat and put your API key. Then save and open the .bat file again.


## Evaluation:
Evaluation shows, that increasing the number of candidate items initially improves performance, but beyond a certain threshold introduces noise and degrades recommendation quality.
The best performance was achieved with approximately 5000 candidate items, highlighting the importance of effective candidate selection in recommendation systems.

### Evaluation results for 5000 candidates

| Mode           | HitRate@5 | HitRate@10 | Precision@10 |
|----------------|----------:|-----------:|-------------:|
| Content-based  | 0.040     | 0.060      | 0.006        |
| Nearest-user   | 0.320     | 0.400      | 0.044        |
| Hybrid         | 0.160     | 0.300      | 0.032        |

- hitRate@K evaluates whether the recommender system is able to include at least one relevant item within the top-K recommended items for a given user.
- For each user we hide at least one movie that he liked and then we generate recommendations using different modes. Finally, we check if the hidden movie is in the top-K recommended movies.
If it is, then hit=1, otherwise hit=0
- hitRate@K = (# users with at least 1 hit in top-K) / (total users).

## Future Improvements:
- Model optimization, so it do not take so much time for getting results
- Save user profiles and personal movie taste

## Demo screenshots:
<img width="1412" height="738" alt="image" src="https://github.com/user-attachments/assets/562c26bf-a842-4699-bb98-c970fc2ec5d7" />
<img width="1418" height="732" alt="image" src="https://github.com/user-attachments/assets/f9302d9b-d9fc-4c4e-a66e-671d71452d45" />
<img width="1431" height="840" alt="image" src="https://github.com/user-attachments/assets/6f5cb92c-cb41-4069-b923-eccb281822dc" />

## Author
Dimitar Handzhiev, Computer Science @ RWTH Aachen University


