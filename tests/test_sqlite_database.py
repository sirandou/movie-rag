import sqlite3

import pandas as pd
import pytest

from src.data.sqlite_database import MovieDatabase


@pytest.fixture
def db(tmp_path):
    """Connected MovieDatabase backed by a temp file."""
    instance = MovieDatabase(db_path=str(tmp_path / "test.db"))
    instance.connect()
    yield instance
    instance.close()


@pytest.fixture
def db_with_table(db):
    db.create_tables()
    return db


@pytest.fixture
def db_with_movies(db_with_table):
    db_with_table.conn.execute("""
        INSERT INTO movies (rotten_tomatoes_link, movie_title, release_year, imdb_rating, genres)
        VALUES
            ('/m/inception',    'Inception',    2010, 8.8, 'Sci-Fi'),
            ('/m/matrix',       'The Matrix',   1999, 8.7, 'Action'),
            ('/m/pulp_fiction', 'Pulp Fiction', 1994, 8.9, 'Crime')
    """)
    db_with_table.conn.commit()
    return db_with_table


@pytest.fixture
def minimal_movies_csv(tmp_path):
    """Minimal CSV matching the MovieDatabase schema."""
    df = pd.DataFrame(
        {
            "rotten_tomatoes_link": ["/m/a", "/m/b", "/m/a"],  # duplicate to test dedup
            "movie_title": ["Movie A", "Movie B", "Movie A"],
            "original_release_date": ["2010-01-01", "2015-05-15", "2010-01-01"],
            "imdb_rating": [7.5, 8.0, 7.5],
            "tomatometer_rating": [80.0, 90.0, 80.0],
            "audience_rating": [70.0, 85.0, 70.0],
            "runtime": [120, 110, 120],
            "tomatometer_count": [100, 200, 100],
            "audience_count": [5000, 8000, 5000],
            "tomatometer_top_critics_count": [50, 100, 50],
            "tomatometer_fresh_critics_count": [80, 180, 80],
            "tomatometer_rotten_critics_count": [20, 20, 20],
            "genres": ["Drama", "Comedy", "Drama"],
            "directors": ["Dir A", "Dir B", "Dir A"],
            "actors": ["Actor A", "Actor B", "Actor A"],
            "authors": [None, None, None],
            "content_rating": ["PG-13", "PG", "PG-13"],
            "awards": [None, None, None],
            "box_office": [None, None, None],
            "production_company": ["Studio A", "Studio B", "Studio A"],
            "tomatometer_status": ["Fresh", "Certified Fresh", "Fresh"],
            "audience_status": ["Upright", "Upright", "Upright"],
            "streaming_release_date": [None, None, None],
        }
    )
    path = tmp_path / "movies.csv"
    df.to_csv(path, index=False)
    return str(path)


# --- connect ---


def test_connect_returns_sqlite_connection(tmp_path):
    instance = MovieDatabase(db_path=str(tmp_path / "test.db"))
    conn = instance.connect()
    assert isinstance(conn, sqlite3.Connection)
    instance.close()


def test_connect_creates_db_file(tmp_path):
    path = tmp_path / "test.db"
    instance = MovieDatabase(db_path=str(path))
    instance.connect()
    assert path.exists()
    instance.close()


# --- create_tables ---


def test_create_tables_movies_table_exists(db_with_table):
    tables = db_with_table.query("SELECT name FROM sqlite_master WHERE type='table'")
    assert "movies" in tables["name"].values


def test_create_tables_indexes_created(db_with_table):
    indexes = db_with_table.query("SELECT name FROM sqlite_master WHERE type='index'")
    for expected in ("idx_imdb_rating", "idx_release_year", "idx_tomatometer_rating"):
        assert expected in indexes["name"].values


def test_create_tables_idempotent(db_with_table):
    db_with_table.create_tables()  # calling twice should not raise
    tables = db_with_table.query("SELECT name FROM sqlite_master WHERE type='table'")
    assert "movies" in tables["name"].values


# --- query ---


def test_query_returns_dataframe(db_with_movies):
    result = db_with_movies.query("SELECT * FROM movies")
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3


def test_query_filter_by_rating(db_with_movies):
    result = db_with_movies.query("SELECT * FROM movies WHERE imdb_rating > 8.8")
    assert len(result) == 1
    assert result.iloc[0]["movie_title"] == "Pulp Fiction"


def test_query_ordering(db_with_movies):
    result = db_with_movies.query(
        "SELECT movie_title FROM movies ORDER BY imdb_rating DESC"
    )
    assert result.iloc[0]["movie_title"] == "Pulp Fiction"


def test_query_count(db_with_movies):
    result = db_with_movies.query("SELECT COUNT(*) as n FROM movies")
    assert result.iloc[0]["n"] == 3


def test_query_empty_result(db_with_movies):
    result = db_with_movies.query("SELECT * FROM movies WHERE imdb_rating > 10")
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


# --- load_from_csv ---


def test_load_from_csv_inserts_rows(db_with_table, minimal_movies_csv):
    db_with_table.load_from_csv(minimal_movies_csv)
    result = db_with_table.query("SELECT COUNT(*) as n FROM movies")
    # 3 CSV rows but 2 unique rotten_tomatoes_link values — deduped by groupby().first()
    assert result.iloc[0]["n"] == 2


def test_load_from_csv_derives_release_year(db_with_table, minimal_movies_csv):
    db_with_table.load_from_csv(minimal_movies_csv)
    result = db_with_table.query(
        "SELECT release_year FROM movies WHERE movie_title='Movie A'"
    )
    assert result.iloc[0]["release_year"] == 2010


# --- close ---


def test_close_can_be_called_twice(tmp_path):
    instance = MovieDatabase(db_path=str(tmp_path / "test.db"))
    instance.connect()
    instance.close()
    instance.close()  # should not raise
