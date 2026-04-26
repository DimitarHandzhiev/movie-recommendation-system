import { useDeferredValue, useEffect, useState} from "react";
import StarRating from "../components/StarRatings";
import { getSeedMovies, replaceMovie } from "../services/api";
import { getRecommendations } from "../services/api";

function HomePage(){
    const [mode, setMode] = useState("content");
    const [movies, setMovies] = useState([]);
    const [ratings, setRatings] = useState({});
    const [recommendations, setRecommendations] = useState([]);
    const [isLoadingMovies, setIsLoadingMovies] = useState(false);
    const [isLoadingRecommendations, setIsLoadingRecommendations] = useState(false);

    useEffect(() => {
        loadMovies();
    }, []);

    const loadMovies = async () => {
        setIsLoadingMovies(true);

        try{
            const data = await getSeedMovies();
            setMovies(data);
        } catch (err) {
            console.error(err);
        } finally {
            setIsLoadingMovies(false);
        }
    };

    const handleRatingChange = (movieId, value) => {
        setRatings((prev) => ({
            ...prev,
            [movieId]: value === "" ? "" : parseFloat(value),
        }));
    };

    const handleReplaceMovie = async (movieIdToReplace) => {
        try{
            const excludeIds = movies.map((movie) => movie.movieId);
            const newMovie = await replaceMovie(excludeIds);

            if(newMovie.error) {
                console.error(newMovie.error);
                return;
            }
            
            setMovies((prevMovies) => prevMovies.map((movie)=> movie.movieId === movieIdToReplace ? newMovie : movie));
            setRatings((prevRatings)=> {const updatedRatings = {...prevRatings}; delete updatedRatings[movieIdToReplace]; return updatedRatings;});
        } catch (error){
            console.error("Replace movie failed:", error);
        }
    };

    const handleRecommend = async () => {
        setIsLoadingRecommendations(true);

        try {
            const result = await getRecommendations(mode, ratings);

            if (Array.isArray(result)) {
                setRecommendations(result);
            } else if (result.error) {
                alert(result.error);
                setRecommendations([]);
            } else if (result.detail) {
                alert(JSON.stringify(result.detail, null, 2));
                setRecommendations([]);
            } else {
                alert(JSON.stringify(result, null, 2));
                setRecommendations([]);
            }
        } catch (error){
            console.error(error);
            alert("Failed to get recommendations");
        } finally {
            setIsLoadingRecommendations(false);
        }        
    };

    const LoadingSpinner = ({ text }) => (
        <div className="loader">⏳ {text} </div>
    );


    return (
    <div className="app-container">
      <div className="app-card">
        <h1 className="app-title">Movie Recommendation System</h1>
        <p className="app-subtitle">
          Compare content-based, nearest-user and hybrid recommendation models.
        </p>

        <label>Select mode: </label>
        <select
          className="mode-select"
          value={mode}
          onChange={(e) => setMode(e.target.value)}>
          <option value="content">Content-Based Recommender</option>
          <option value="nearest">Nearest-User Recommender</option>
          <option value="hybrid">Hybrid Recommender</option>
        </select>

        <p>Current mode: {mode}</p>

        <h2 className="section-title">Rate these movies:</h2>

        {isLoadingMovies && <LoadingSpinner text="Loading seed movies..." />}

        {!isLoadingMovies && (
        <div className="seed-grid">{movies.map((movie) => (
            <div key={movie.movieId} className="movie-card">
                <div>
                <p className="movie-title">{movie.title}</p>
                <div className="movie-rating-area">
                    <StarRating rating={ratings[movie.movieId] !== undefined ? parseFloat(ratings[movie.movieId]) : 0}onRatingChange={(value) => handleRatingChange(movie.movieId, value.toString())}/>
                </div>
                </div>
                <div className="movie-actions">
                <button className="secondary-button" onClick={() => handleReplaceMovie(movie.movieId)}>Not Watched</button>
                </div>
            </div>
            ))}
        </div>
        )}

        <h3 className="section-title">Current ratings:</h3>
        <pre className="rating-box">{JSON.stringify(ratings, null, 2)}</pre>

        <button className="primary-button" onClick={handleRecommend} disabled={isLoadingRecommendations}>
          {isLoadingRecommendations ? "Generating..." : "Get Recommendations"}
        </button>

        {isLoadingRecommendations && (<LoadingSpinner text="Generating recommendations..." />)}

        <h2 className="section-title">Recommendations:</h2>

        {Array.isArray(recommendations) && recommendations.length > 0 && (<div className="recommendation-grid">{recommendations.map((m) => (
                <div key={m.movieId} className="recommendation-card">
                    <div className="recommendation-title">{m.title}</div>
                    {m.genres && (<div className="genre-tags">{m.genres.split("|").map((g)=>(<span key={g} className="genre-tag">{g}</span>))}</div>)}
                </div>
                ))}
            </div>
        )}
      </div>
    </div>
  );
}

export default HomePage;