import numpy as np
import pandas as pd
import pytest

from src.data.document_creators import create_plot_docs, create_poster_docs, create_review_docs


@pytest.fixture
def plots_df():
    return pd.DataFrame(
        {
            "movie_title": ["Inception", "The Matrix", "Interstellar"],
            "release_year": [2010, 1999, 2014],
            "directors": ["Nolan", "Wachowski", "Nolan"],
            "genres": ["Sci-Fi", "Action", "Sci-Fi"],
            "plot": [
                "A thief who steals corporate secrets through dreams.",
                "A hacker discovers reality is a simulation.",
                None,  # should be dropped
            ],
        }
    )


@pytest.fixture
def reviews_df():
    return pd.DataFrame(
        {
            "rotten_tomatoes_link": ["/m/inception", "/m/inception", "/m/matrix"],
            "movie_title": ["Inception", "Inception", "The Matrix"],
            "release_year": [2010, 2010, 1999],
            "review_content": ["Mind-bending.", "A masterpiece.", None],
            "directors": ["Nolan", "Nolan", "Wachowski"],
        }
    )


@pytest.fixture
def posters_df():
    return pd.DataFrame(
        {
            "movie_title": ["Inception", "The Matrix", "Missing"],
            "release_year": [2010, 1999, 2000],
            "genres": ["Sci-Fi", "Action", "Drama"],
            "directors": ["Nolan", "Wachowski", "Unknown"],
            "actors": ["DiCaprio,Gordon-Levitt,Page", "Reeves,Fishburne", "Actor A"],
            "poster_path": ["/posters/inception.jpg", "/posters/matrix.jpg", None],
        }
    )


# --- create_plot_docs ---


def test_create_plot_docs_returns_list(plots_df):
    docs = create_plot_docs(plots_df, [], [])
    assert isinstance(docs, list)


def test_create_plot_docs_drops_nan_plot(plots_df):
    docs = create_plot_docs(plots_df, [], [])
    assert len(docs) == 2  # Interstellar has NaN plot


def test_create_plot_docs_structure(plots_df):
    docs = create_plot_docs(plots_df, [], [])
    for doc in docs:
        assert "page_content" in doc
        assert "metadata" in doc
        assert doc["metadata"]["source"] == "plot"


def test_create_plot_docs_text_metadata_prepended(plots_df):
    docs = create_plot_docs(plots_df, ["movie_title", "directors"], [])
    assert "Movie title:" in docs[0]["page_content"]
    assert "Directors:" in docs[0]["page_content"]


def test_create_plot_docs_obj_metadata_in_metadata(plots_df):
    docs = create_plot_docs(plots_df, [], ["movie_title", "release_year"])
    assert docs[0]["metadata"]["movie_title"] in ("Inception", "The Matrix")
    assert "release_year" in docs[0]["metadata"]


def test_create_plot_docs_plot_in_content(plots_df):
    docs = create_plot_docs(plots_df, [], [])
    assert "Plot:" in docs[0]["page_content"]


# --- create_review_docs ---


def test_create_review_docs_returns_list(reviews_df):
    docs = create_review_docs(
        reviews_df, [], [], ["rotten_tomatoes_link", "movie_title", "release_year"]
    )
    assert isinstance(docs, list)


def test_create_review_docs_drops_nan_reviews(reviews_df):
    docs = create_review_docs(
        reviews_df,
        [],
        ["movie_title"],
        ["rotten_tomatoes_link", "movie_title", "release_year"],
    )
    # The Matrix row has NaN review_content, so only Inception group survives
    titles = {doc["metadata"].get("movie_title") for doc in docs}
    assert "Inception" in titles
    assert "The Matrix" not in titles


def test_create_review_docs_aggregates_by_movie(reviews_df):
    docs = create_review_docs(
        reviews_df,
        [],
        ["movie_title"],
        ["rotten_tomatoes_link", "movie_title", "release_year"],
    )
    inception_doc = next(d for d in docs if d["metadata"].get("movie_title") == "Inception")
    assert "Mind-bending." in inception_doc["page_content"]
    assert "A masterpiece." in inception_doc["page_content"]


def test_create_review_docs_structure(reviews_df):
    docs = create_review_docs(
        reviews_df, [], [], ["rotten_tomatoes_link", "movie_title", "release_year"]
    )
    for doc in docs:
        assert "page_content" in doc
        assert "metadata" in doc
        assert doc["metadata"]["source"] == "review"


# --- create_poster_docs ---


def test_create_poster_docs_returns_list(posters_df):
    docs = create_poster_docs(posters_df, [])
    assert isinstance(docs, list)


def test_create_poster_docs_drops_nan_poster_path(posters_df):
    docs = create_poster_docs(posters_df, [])
    assert len(docs) == 2  # "Missing" has NaN poster_path


def test_create_poster_docs_structure(posters_df):
    docs = create_poster_docs(posters_df, [])
    for doc in docs:
        assert "poster_path" in doc
        assert "text_content" in doc
        assert "metadata" in doc
        assert doc["metadata"]["source"] == "poster"


def test_create_poster_docs_obj_metadata(posters_df):
    docs = create_poster_docs(posters_df, ["movie_title", "release_year"])
    assert "movie_title" in docs[0]["metadata"]
    assert "release_year" in docs[0]["metadata"]


def test_create_poster_docs_truncates_long_text():
    long_actors = ",".join([f"Actor{i}" for i in range(100)])
    df = pd.DataFrame(
        {
            "movie_title": ["Very Long Film Title That Goes On And On"],
            "release_year": [2020],
            "genres": ["Drama, Comedy, Romance, Thriller, Action, Sci-Fi, Horror, Mystery"],
            "directors": ["Director One, Director Two, Director Three"],
            "actors": [long_actors],
            "poster_path": ["/poster.jpg"],
        }
    )
    docs = create_poster_docs(df, [])
    words = docs[0]["text_content"].split()
    assert len(words) <= 62  # 60 words + "..." possible trailing
