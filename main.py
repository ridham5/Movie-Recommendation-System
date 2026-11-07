"""Content-based movie recommender: finds movies with the most similar genres.

Usage:
    python main.py "Toy Story (1995)"
    python main.py "Heat (1995)" -n 10 --movies movies_dataset/movie.csv
"""

import argparse

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_MOVIES = "data/sample_movies.csv"


def load_movies(path):
    """Load a MovieLens-style CSV with movieId, title and genres columns (one row per movie)."""
    movies = pd.read_csv(path, usecols=["movieId", "title", "genres"])
    movies = movies.drop_duplicates(subset="title").reset_index(drop=True)
    movies = movies[movies["genres"] != "(no genres listed)"].reset_index(drop=True)
    return movies


def build_similarity(movies):
    """Turn each movie's genres into a vector, then compare every pair with cosine similarity."""
    vectorizer = CountVectorizer(tokenizer=lambda g: g.split("|"), token_pattern=None, lowercase=False)
    genre_matrix = vectorizer.fit_transform(movies["genres"])
    return cosine_similarity(genre_matrix)


def recommend(movies, similarity, title, n=5):
    """Return the n movies whose genres are most similar to `title`."""
    matches = movies.index[movies["title"] == title]
    if len(matches) == 0:
        raise KeyError(f"Movie not found: {title}")
    idx = matches[0]

    scores = sorted(enumerate(similarity[idx]), key=lambda pair: pair[1], reverse=True)
    top = [i for i, _ in scores if i != idx][:n]

    result = movies.iloc[top][["title", "genres"]].copy()
    result["similarity"] = [round(similarity[idx][i], 2) for i in top]
    return result.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Recommend movies with similar genres.")
    parser.add_argument("title", nargs="?", default="Toy Story (1995)")
    parser.add_argument("-n", type=int, default=5, help="number of recommendations")
    parser.add_argument("--movies", default=DEFAULT_MOVIES, help="path to movies CSV")
    args = parser.parse_args()

    movies = load_movies(args.movies)
    similarity = build_similarity(movies)
    try:
        print(f"\nMovies similar to '{args.title}':\n")
        print(recommend(movies, similarity, args.title, args.n).to_string(index=False))
    except KeyError as err:
        print(err)


if __name__ == "__main__":
    main()
