function RecommendationList({ movies }) {
  return (
    <div>
      <h2>Recommendations</h2>
      {movies.map((m) => (
        <p key={m.movieId}>{m.title}</p>
      ))}
    </div>
  );
}

export default RecommendationList;