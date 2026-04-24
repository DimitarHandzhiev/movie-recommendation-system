import { useDeferredValue, useEffect, useState} from "react";
import StarRating from "../components/StarRatings";
import { getSeedMovies, replaceMovie } from "../services/api";
import { getRecommendations } from "../services/api";

function HomePage(){
    const [mode, setMode] = useState("content");
    const [movies, setMovies] = useState([]);
    const [ratings, setRatings] = useState({});
    const [recommendations, setRecommendations] = useState([]);

    useEffect(() => {
        loadMovies();
    }, []);

    const loadMovies = async () => {
        try{
            const data = await getSeedMovies();
            setMovies(data);
        } catch (err) {
            console.error(err);
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
        const result = await getRecommendations(mode, ratings);

        console.log("Recommendation result:", result);

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
    };


    return(
        <div style={{padding: "2rem"}}>
            <h1>Movie Recommendation System</h1>

            <label>Select mode: </label>
            <select value={mode} onChange={(e) => setMode(e.target.value)}>
                <option value="content">Content-Based Recommender</option>
                <option value="nearest">Nearest-User Recommender</option>
                <option value="hybrid">Hybrid Recommender</option>
            </select>

            <p>Current mode: {mode}</p>

            <h2>Rate these movies:</h2>
            {movies.map((movie)=>(
                <div key={movie.movieId} style={{marginBottom: "1rem", padding: "1rem", border: "1px solid #ccc", borderRadius: "8px",}}>
                    <p><strong>{movie.title}</strong></p>
                <StarRating rating={ratings[movie.movieId] ? parseFloat(ratings[movie.movieId]) : 0} onRatingChange={(value) => handleRatingChange(movie.movieId, value.toString())}/>
                <button onClick={()=> handleReplaceMovie(movie.movieId)} style={{marginTop: "0.5rem", padding:"0.4rem 0.8rem", borderRadius:"6px", border:"1px solid #ccc", cursor:"pointer", background:"#f8f8f8",}}>
                    Not Watched
                </button>
            </div>
            ))}
            <h3>Current ratings:</h3>
            <pre>{JSON.stringify(ratings, null, 2)}</pre>
            <button onClick={handleRecommend}>Get Recommendations</button>
            <h2>Recommendations:</h2>
            {Array.isArray(recommendations) &&
                recommendations.map((m) => (
                <p key={m.movieId}>{m.title}</p>
                ))}
        </div>
    );
}

export default HomePage;