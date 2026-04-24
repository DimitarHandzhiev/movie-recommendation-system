const BASE_URL = "http://127.0.0.1:8000";

export async function getHealth() {
  const response = await fetch(`${BASE_URL}/health`);
  if (!response.ok) {
    throw new Error(`HTTP error: ${response.status}`);
  }
  return await response.json();
}

export async function getSeedMovies() {
    const response = await fetch("http://localhost:8000/seed-movies");
    if(!response.ok){
        throw new Error("Failed to fetch seed movies");
    }
    return response.json();
}

export async function replaceMovie(excludeMovieIds) {
    const response = await fetch("http://localhost:8000/replace-movie", {method: "POST", headers:{"Content-Type": "application/json",}, body: JSON.stringify({exclude_movie_ids: excludeMovieIds,}),});

    if(!response.ok){
        throw new Error("Failed to replace movie");
    }

    return response.json();
}

export async function getRecommendations(mode, ratings) {
    const response = await fetch("http://127.0.0.1:8000/recommend",{method:"POST",headers:{"Content-Type":"application/json",}, body: JSON.stringify({mode:mode,ratings:ratings,top_n:10,}),});
    return await response.json();
}