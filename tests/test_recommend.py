import pytest

from main import build_similarity, load_movies, recommend


@pytest.fixture(scope="module")
def model():
    movies = load_movies("data/sample_movies.csv")
    return movies, build_similarity(movies)


def test_toy_story_gets_animated_family_movies(model):
    movies, sim = model
    recs = recommend(movies, sim, "Toy Story (1995)", n=3)
    assert len(recs) == 3
    assert all("Animation" in g for g in recs["genres"])
    assert recs["similarity"].iloc[0] == 1.0  # Antz / Toy Story 2 / Monsters, Inc. have identical genres


def test_never_recommends_the_same_movie(model):
    movies, sim = model
    recs = recommend(movies, sim, "Heat (1995)", n=10)
    assert "Heat (1995)" not in recs["title"].tolist()
    assert recs["title"].is_unique


def test_results_sorted_by_similarity(model):
    movies, sim = model
    scores = recommend(movies, sim, "Matrix, The (1999)", n=5)["similarity"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_unknown_title_raises(model):
    movies, sim = model
    with pytest.raises(KeyError):
        recommend(movies, sim, "Not A Real Movie (2099)")
