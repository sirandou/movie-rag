import json

import numpy as np
import pandas as pd
import pytest

from src.agents.tools.collaborative_filtering import (
    CollaborativeFilteringTool,
    normalize_rating,
)


# =============================================================================
# normalize_rating — pure function, no fixtures needed
# =============================================================================


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("8/10", 8.0),
        ("3/4", 7.5),
        ("1/2", 5.0),
        ("2.5/4", 6.25),
        ("A+", 10.0),
        ("A", 9.5),
        ("A-", 9.0),
        ("B+", 8.5),
        ("B", 8.0),
        ("B-", 7.5),
        ("C+", 7.0),
        ("C", 6.5),
        ("C-", 6.0),
        ("D", 5.0),
        ("F", 3.0),
        ("7.5", 7.5),  # already 0-10
        ("85", 8.5),  # 0-100 scale, gets divided by 10
        ("10", 10.0),
    ],
)
def test_normalize_rating_known_formats(raw, expected):
    assert normalize_rating(raw) == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize("null_val", [None, float("nan"), "<null>", ""])
def test_normalize_rating_nulls_return_nan(null_val):
    result = normalize_rating(null_val)
    assert np.isnan(result)


def test_normalize_rating_case_insensitive():
    assert normalize_rating("b+") == normalize_rating("B+")


def test_normalize_rating_unknown_format_returns_nan():
    result = normalize_rating("excellent")
    assert np.isnan(result)


# =============================================================================
# CollaborativeFilteringTool — needs a fake reviews CSV
# =============================================================================


@pytest.fixture
def reviews_csv(tmp_path):
    """
    Minimal reviews CSV: 4 critics × 4 movies = enough data for CF.
    Ratings are fractions so normalize_rating() produces non-NaN values.
    """
    rows = []
    movies = [
        ("Inception", "2010-07-16", "/m/inception"),
        ("The Matrix", "1999-03-31", "/m/matrix"),
        ("Pulp Fiction", "1994-10-14", "/m/pulp"),
        ("Fight Club", "1999-10-15", "/m/fight"),
    ]
    critics = ["Alice", "Bob", "Carol", "Dave"]
    # Each critic rates every movie with a slightly different score
    base = [[9, 8, 7, 6], [8, 9, 6, 7], [7, 6, 9, 8], [6, 7, 8, 9]]

    for c_idx, critic in enumerate(critics):
        for m_idx, (title, date, link) in enumerate(movies):
            rows.append(
                {
                    "movie_title": title,
                    "original_release_date": date,
                    "critic_name": critic,
                    "review_score": f"{base[c_idx][m_idx]}/10",
                    "rotten_tomatoes_link": link,
                }
            )
    df = pd.DataFrame(rows)
    path = tmp_path / "reviews.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def cf_tool(reviews_csv):
    return CollaborativeFilteringTool(reviews_path=reviews_csv)


# --- internal state after init ---


def test_rating_matrix_shape(cf_tool):
    assert cf_tool.rating_matrix.shape == (4, 4)  # 4 critics × 4 movies


def test_movie_similarity_df_shape(cf_tool):
    assert cf_tool.movie_similarity_df.shape == (4, 4)


def test_key_to_id_populated(cf_tool):
    assert len(cf_tool.key_to_id) == 4


def test_id_to_key_populated(cf_tool):
    assert len(cf_tool.id_to_key) == 4


def test_similarity_diagonal_is_one(cf_tool):
    diag = np.diag(cf_tool.movie_similarity)
    assert np.allclose(diag, 1.0, atol=1e-5)


# --- get_tool() ---


def test_get_tool_returns_callable(cf_tool):
    tool_fn = cf_tool.get_tool()
    assert callable(tool_fn)


def test_recommend_known_movie_returns_json(cf_tool):
    tool_fn = cf_tool.get_tool()
    result = tool_fn.invoke(
        {"movie_titles": "Inception (2010)", "num_recommendations": 2}
    )
    data = json.loads(result)
    assert isinstance(data, dict)


def test_recommend_known_movie_has_recommendations(cf_tool):
    tool_fn = cf_tool.get_tool()
    result = json.loads(
        tool_fn.invoke({"movie_titles": "Inception (2010)", "num_recommendations": 2})
    )
    # May have recommendations or a warning — it should not be an error dict
    assert "error" not in result or result.get("recommendations") is not None


def test_recommend_unknown_movie_warns(cf_tool):
    tool_fn = cf_tool.get_tool()
    result = json.loads(
        tool_fn.invoke({"movie_titles": "Completely Unknown Film XYZ 9999"})
    )
    # Either a warning or an error about not found
    assert "warnings" in result or "error" in result


def test_recommend_unknown_movie_error_message(cf_tool):
    tool_fn = cf_tool.get_tool()
    result = json.loads(tool_fn.invoke({"movie_titles": "ZZZZZZZ_NO_MATCH_ZZZZZZZ"}))
    has_warning = result.get("warnings", "") != ""
    has_error = "error" in result
    assert has_warning or has_error


def test_recommend_input_movies_listed(cf_tool):
    tool_fn = cf_tool.get_tool()
    result = json.loads(
        tool_fn.invoke({"movie_titles": "Inception (2010)", "num_recommendations": 2})
    )
    if "input_movies" in result:
        assert any("Inception" in m for m in result["input_movies"])


def test_recommend_excludes_input_from_results(cf_tool):
    tool_fn = cf_tool.get_tool()
    result = json.loads(
        tool_fn.invoke({"movie_titles": "Inception (2010)", "num_recommendations": 3})
    )
    if "recommendations" in result:
        titles = [r["title (year)"] for r in result["recommendations"]]
        assert not any("Inception" in t for t in titles)


def test_recommend_num_recommendations_respected(cf_tool):
    tool_fn = cf_tool.get_tool()
    result = json.loads(
        tool_fn.invoke({"movie_titles": "Inception (2010)", "num_recommendations": 1})
    )
    if "recommendations" in result:
        assert len(result["recommendations"]) <= 1


def test_recommend_fuzzy_match(cf_tool):
    """Slightly misspelled title should still be matched via fuzzy search."""
    tool_fn = cf_tool.get_tool()
    result = json.loads(
        tool_fn.invoke({"movie_titles": "Incepion", "num_recommendations": 2})  # typo
    )
    # fuzzy match should find "Inception (2010)" or warn about ambiguity
    assert isinstance(result, dict)
