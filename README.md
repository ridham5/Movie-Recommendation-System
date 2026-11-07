# Movie Recommendation System

![CI](https://github.com/ridham5/Movie-Recommendation-System/actions/workflows/ci.yml/badge.svg)

A content-based movie recommender in Python. Give it a movie and it returns the movies with the
most similar genres, ranked by cosine similarity.

## How it works

1. Each movie's genres (for example `Adventure|Animation|Children`) become a vector with
   scikit-learn's `CountVectorizer`.
2. Cosine similarity compares every pair of movies.
3. For the chosen movie, the top-N most similar movies are returned (never the movie itself).

## Example

```
$ python main.py "Toy Story (1995)"

                title                                              genres  similarity
          Antz (1998)         Adventure|Animation|Children|Comedy|Fantasy        1.00
   Toy Story 2 (1999)         Adventure|Animation|Children|Comedy|Fantasy        1.00
Monsters, Inc. (2001)         Adventure|Animation|Children|Comedy|Fantasy        1.00
         Shrek (2001) Adventure|Animation|Children|Comedy|Fantasy|Romance        0.91
 Bug's Life, A (1998)                 Adventure|Animation|Children|Comedy        0.89
```

## Run it

```bash
pip install -r requirements.txt
python main.py "Toy Story (1995)"          # uses the small sample in data/
python main.py "Heat (1995)" -n 10
pytest -v
```

To use the full [MovieLens 20M dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset),
download it, put `movie.csv` in `movies_dataset/`, and run:

```bash
python main.py "Toy Story (1995)" --movies movies_dataset/movie.csv
```

## Tech

Python · pandas · scikit-learn · pytest · GitHub Actions

## Next steps

- Blend in user ratings (collaborative filtering) for a hybrid recommender
- Serve recommendations through a small REST API
