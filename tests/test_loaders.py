import pandas as pd
import pytest
from langchain.schema import Document

from src.langchain.loaders import MovieTextDocumentLoader


# ---------------------------------------------------------------------------
# Fixtures: minimal CSVs that satisfy the loader's expected schema
# ---------------------------------------------------------------------------


@pytest.fixture
def plots_csv(tmp_path):
    df = pd.DataFrame(
        {
            "movie_title": ["Inception", "The Matrix"],
            "original_release_date": ["2010-07-16", "1999-03-31"],
            "plot": [
                "A thief who enters dreams to steal secrets.",
                "A hacker discovers reality is a simulation.",
            ],
            "rotten_tomatoes_link": ["/m/inception", "/m/matrix"],
            "directors": ["Christopher Nolan", "The Wachowskis"],
            "genres": ["Sci-Fi, Thriller", "Action, Sci-Fi"],
        }
    )
    path = tmp_path / "plots.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def reviews_csv(tmp_path):
    df = pd.DataFrame(
        {
            "movie_title": ["Inception", "Inception", "The Matrix"],
            "original_release_date": ["2010-07-16", "2010-07-16", "1999-03-31"],
            "review_content": ["Mind-bending.", "A masterpiece.", "A classic."],
            "rotten_tomatoes_link": ["/m/inception", "/m/inception", "/m/matrix"],
            "critic_name": ["Critic A", "Critic B", "Critic A"],
            "directors": ["Nolan", "Nolan", "Wachowskis"],
            "genres": ["Sci-Fi", "Sci-Fi", "Action"],
        }
    )
    path = tmp_path / "reviews.csv"
    df.to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# Plots-only loading
# ---------------------------------------------------------------------------


def test_load_plots_returns_documents(plots_csv):
    loader = MovieTextDocumentLoader(plots_path=plots_csv)
    docs = loader.load()
    assert isinstance(docs, list)
    assert len(docs) > 0
    assert all(isinstance(d, Document) for d in docs)


def test_load_plots_count(plots_csv):
    loader = MovieTextDocumentLoader(plots_path=plots_csv)
    docs = loader.load()
    assert len(docs) == 2  # two movies with valid plots


def test_load_plots_page_content_non_empty(plots_csv):
    loader = MovieTextDocumentLoader(plots_path=plots_csv)
    docs = loader.load()
    assert all(d.page_content for d in docs)


def test_load_plots_metadata_source(plots_csv):
    loader = MovieTextDocumentLoader(plots_path=plots_csv)
    docs = loader.load()
    assert all(d.metadata.get("source") == "plot" for d in docs)


def test_load_plots_contains_plot_text(plots_csv):
    loader = MovieTextDocumentLoader(plots_path=plots_csv)
    docs = loader.load()
    combined = " ".join(d.page_content for d in docs)
    assert "dreams" in combined or "simulation" in combined


# ---------------------------------------------------------------------------
# Reviews-only loading
# ---------------------------------------------------------------------------


def test_load_reviews_returns_documents(reviews_csv):
    loader = MovieTextDocumentLoader(reviews_path=reviews_csv)
    docs = loader.load()
    assert isinstance(docs, list)
    assert len(docs) > 0


def test_load_reviews_aggregated_per_movie(reviews_csv):
    loader = MovieTextDocumentLoader(reviews_path=reviews_csv)
    docs = loader.load()
    # Two unique movies; Inception has 2 reviews aggregated into 1 doc
    assert len(docs) == 2


def test_load_reviews_metadata_source(reviews_csv):
    loader = MovieTextDocumentLoader(reviews_path=reviews_csv)
    docs = loader.load()
    assert all(d.metadata.get("source") == "review" for d in docs)


# ---------------------------------------------------------------------------
# Combined loading
# ---------------------------------------------------------------------------


def test_load_both_combines_docs(plots_csv, reviews_csv):
    loader = MovieTextDocumentLoader(plots_path=plots_csv, reviews_path=reviews_csv)
    docs = loader.load()
    sources = {d.metadata.get("source") for d in docs}
    assert "plot" in sources
    assert "review" in sources


def test_load_both_count(plots_csv, reviews_csv):
    loader = MovieTextDocumentLoader(plots_path=plots_csv, reviews_path=reviews_csv)
    docs = loader.load()
    assert len(docs) == 4  # 2 plots + 2 review groups


# ---------------------------------------------------------------------------
# lazy_load
# ---------------------------------------------------------------------------


def test_lazy_load_is_iterator(plots_csv):
    loader = MovieTextDocumentLoader(plots_path=plots_csv)
    it = loader.lazy_load()
    # Must be iterable
    docs = list(it)
    assert len(docs) > 0
    assert all(isinstance(d, Document) for d in docs)
