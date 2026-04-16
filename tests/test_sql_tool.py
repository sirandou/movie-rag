from unittest.mock import MagicMock, patch

import pytest

from src.agents.tools.sql_tool import SQLMovieTool
from src.data.sqlite_database import MovieDatabase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path):
    """Real SQLite DB pre-populated with movie rows."""
    path = str(tmp_path / "movies.db")
    db = MovieDatabase(db_path=path)
    db.connect()
    db.create_tables()
    db.conn.execute("""
        INSERT INTO movies (rotten_tomatoes_link, movie_title, release_year, imdb_rating, genres)
        VALUES
            ('/m/inception',    'Inception',    2010, 8.8, 'Sci-Fi,Thriller'),
            ('/m/matrix',       'The Matrix',   1999, 8.7, 'Action,Sci-Fi'),
            ('/m/pulp_fiction', 'Pulp Fiction', 1994, 8.9, 'Crime,Drama')
    """)
    db.conn.commit()
    db.close()
    return path


def _make_tool(db_path, sql_to_return: str):
    """Build an SQLMovieTool with the LLM mocked to return a fixed SQL string."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = sql_to_return
    with patch("src.agents.tools.sql_tool.ChatOpenAI", return_value=mock_llm):
        tool_obj = SQLMovieTool(db_path=db_path)
    return tool_obj, mock_llm


# ---------------------------------------------------------------------------
# get_tool() returns a callable tool
# ---------------------------------------------------------------------------


def test_get_tool_returns_callable(db_path):
    tool_obj, _ = _make_tool(db_path, "SELECT * FROM movies")
    tool_fn = tool_obj.get_tool()
    assert callable(tool_fn)


# ---------------------------------------------------------------------------
# Happy-path: valid SQL
# ---------------------------------------------------------------------------


def test_tool_returns_string(db_path):
    tool_obj, _ = _make_tool(db_path, "SELECT movie_title FROM movies")
    tool_fn = tool_obj.get_tool()
    result = tool_fn.invoke({"question": "List all movies"})
    assert isinstance(result, str)


def test_tool_returns_movie_data(db_path):
    tool_obj, _ = _make_tool(
        db_path, "SELECT movie_title FROM movies ORDER BY imdb_rating DESC"
    )
    tool_fn = tool_obj.get_tool()
    result = tool_fn.invoke({"question": "Top movies by rating"})
    assert "Pulp Fiction" in result


def test_tool_strips_markdown_from_sql(db_path):
    """LLM sometimes returns SQL wrapped in markdown code fences."""
    sql_with_fences = "```sql\nSELECT movie_title FROM movies\n```"
    tool_obj, _ = _make_tool(db_path, sql_with_fences)
    tool_fn = tool_obj.get_tool()
    result = tool_fn.invoke({"question": "List movies"})
    # If fences weren't stripped the query would error; a valid result means they were
    assert isinstance(result, str)
    assert "Inception" in result or "Matrix" in result or "Pulp" in result


def test_tool_summarises_large_results(db_path):
    """Results > 20 rows should be truncated with a summary header."""
    # Insert 25 more rows so we exceed the 20-row threshold
    db = MovieDatabase(db_path=db_path)
    db.connect()
    for i in range(25):
        db.conn.execute(
            "INSERT OR IGNORE INTO movies (rotten_tomatoes_link, movie_title) VALUES (?, ?)",
            (f"/m/extra{i}", f"Extra Movie {i}"),
        )
    db.conn.commit()
    db.close()

    tool_obj, _ = _make_tool(db_path, "SELECT * FROM movies")
    tool_fn = tool_obj.get_tool()
    result = tool_fn.invoke({"question": "All movies"})
    assert "Found" in result and "results" in result


# ---------------------------------------------------------------------------
# Empty results
# ---------------------------------------------------------------------------


def test_tool_empty_result_message(db_path):
    tool_obj, _ = _make_tool(db_path, "SELECT * FROM movies WHERE imdb_rating > 99")
    tool_fn = tool_obj.get_tool()
    result = tool_fn.invoke({"question": "Movies rated above 99"})
    assert "No movies found" in result


# ---------------------------------------------------------------------------
# Bad SQL / error handling
# ---------------------------------------------------------------------------


def test_tool_bad_sql_returns_error_string(db_path):
    tool_obj, _ = _make_tool(db_path, "SELECT * FROM nonexistent_table")
    tool_fn = tool_obj.get_tool()
    result = tool_fn.invoke({"question": "Something"})
    assert "Error" in result or "error" in result


def test_tool_bad_column_returns_helpful_message(db_path):
    tool_obj, _ = _make_tool(db_path, "SELECT nonexistent_col FROM movies")
    tool_fn = tool_obj.get_tool()
    result = tool_fn.invoke({"question": "Something"})
    assert "no such column" in result.lower() or "Error" in result


# ---------------------------------------------------------------------------
# LLM is called with the formatted prompt
# ---------------------------------------------------------------------------


def test_llm_receives_question_in_prompt(db_path):
    tool_obj, mock_llm = _make_tool(db_path, "SELECT * FROM movies LIMIT 1")
    tool_fn = tool_obj.get_tool()
    tool_fn.invoke({"question": "How many Sci-Fi movies?"})
    prompt_sent = mock_llm.invoke.call_args[0][0]
    assert "How many Sci-Fi movies?" in prompt_sent
