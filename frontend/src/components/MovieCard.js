function MovieCard({ movie, children }) {
  return (
    <div style={{
      border: "1px solid #ccc",
      padding: "1rem",
      borderRadius: "8px",
      marginBottom: "1rem"
    }}>
      <h3>{movie.title}</h3>
      {children}
    </div>
  );
}

export default MovieCard;