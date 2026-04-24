import { useState } from "react";

function StarRating({ rating = 0, onRatingChange }) {
  const [hoverValue, setHoverValue] = useState(null);

  const displayValue = hoverValue !== null ? hoverValue : rating;

  const getStarType = (starIndex, value) => {
    if (value >= starIndex) return "full";
    if (value >= starIndex - 0.5) return "half";
    return "empty";
  };

  const renderOneStar = (starIndex) => {
    const type = getStarType(starIndex, displayValue);

    return (
      <span
        key={starIndex}
        onMouseMove={(e) => {
          const { left, width } = e.currentTarget.getBoundingClientRect();
          const x = e.clientX - left;
          const newValue = x < width / 2 ? starIndex - 0.5 : starIndex;
          setHoverValue(newValue);
        }}
        onMouseLeave={() => setHoverValue(null)}
        onClick={(e) => {
          const { left, width } = e.currentTarget.getBoundingClientRect();
          const x = e.clientX - left;
          const newValue = x < width / 2 ? starIndex - 0.5 : starIndex;
          onRatingChange(newValue);
        }}
        style={{
          cursor: "pointer",
          fontSize: "2rem",
          marginRight: "4px",
          position: "relative",
          display: "inline-block",
          width: "24px",
          userSelect: "none",
          color: type === "empty" ? "#ccc" : "#f5b301",
        }}
      >
        {type === "half" ? (
          <span style={{ position: "relative", display: "inline-block" }}>
            <span style={{ color: "#ccc" }}>★</span>
            <span
              style={{
                position: "absolute",
                left: 0,
                top: 0,
                width: "50%",
                overflow: "hidden",
                color: "#f5b301",
              }}
            >
              ★
            </span>
          </span>
        ) : (
          "★"
        )}
      </span>
    );
  };

  return <div>{[1, 2, 3, 4, 5].map(renderOneStar)}</div>;
}

export default StarRating;