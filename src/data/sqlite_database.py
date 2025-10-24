import sqlite3
import pandas as pd
from pathlib import Path


class MovieDatabase:
    """
    SQLite database with movie metadata.
    """

    def __init__(
        self,
        db_path: str,
    ) -> None:
        """
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = None

    def connect(self) -> sqlite3.Connection:
        """Connect to database."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return self.conn

    def create_tables(self) -> None:
        """Create movies metadata table."""
        cursor = self.conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS movies (
            -- Primary Key
            rotten_tomatoes_link TEXT PRIMARY KEY,

            -- Core identifiers
            movie_title TEXT NOT NULL,
            release_year INTEGER,

            -- Ratings (main query targets)
            imdb_rating REAL,
            tomatometer_rating REAL,
            audience_rating REAL,

            -- Counts
            runtime REAL,
            tomatometer_count INTEGER,
            audience_count INTEGER,
            tomatometer_top_critics_count INTEGER,
            tomatometer_fresh_critics_count INTEGER,
            tomatometer_rotten_critics_count INTEGER,

            -- Categories
            genres TEXT,
            directors TEXT,
            actors TEXT,
            authors TEXT,

            -- Other metadata
            content_rating TEXT,
            awards TEXT,
            box_office TEXT,
            production_company TEXT,
            tomatometer_status TEXT,
            audience_status TEXT,
            streaming_release_date TEXT
        )
        """)

        # Create indexes for common queries
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_release_year ON movies(release_year)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_imdb_rating ON movies(imdb_rating)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_tomatometer_rating ON movies(tomatometer_rating)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_audience_rating ON movies(audience_rating)"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_genres ON movies(genres)")

        self.conn.commit()
        print("Movies table created with indexes")

    def load_from_csv(self, movies_path: str) -> None:
        """Load from movie reviews CSV, since it is more complete.
        Args:
            movies_path: Path to movie reviews CSV file
        """
        print(f"Loading movies from {movies_path}...")

        df = pd.read_csv(movies_path)

        # Create release_year
        df["release_year"] = pd.to_datetime(df["original_release_date"]).dt.year

        # Select columns for schema
        movies_data = df[
            [
                "rotten_tomatoes_link",  # because not all movies have imdb ids
                "movie_title",
                "release_year",
                "imdb_rating",
                "tomatometer_rating",
                "audience_rating",
                "runtime",
                "tomatometer_count",
                "audience_count",
                "tomatometer_top_critics_count",
                "tomatometer_fresh_critics_count",
                "tomatometer_rotten_critics_count",
                "genres",
                "directors",
                "actors",
                "authors",
                "content_rating",
                "awards",
                "box_office",
                "production_company",
                "tomatometer_status",
                "audience_status",
                "streaming_release_date",
            ]
        ].copy()

        grouped = (
            movies_data.groupby("rotten_tomatoes_link", as_index=False)
            .first()
            .reset_index(drop=True)
        )

        # Insert into database
        grouped.to_sql("movies", self.conn, if_exists="replace", index=False)
        print(f"✓ Loaded {len(grouped)} movies")

        self.conn.commit()

    def query(self, sql: str) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame.

        Args:
            sql: SQL query string

        Returns:
            Query results as DataFrame
        """
        return pd.read_sql_query(sql, self.conn)

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


def setup_database(db_path: str, csv_path: str, verbose: bool = True) -> None:
    """Initialize database with plots CSV."""
    print("=" * 60)
    print("Setting up movie database")
    print("=" * 60)

    db = MovieDatabase(db_path=db_path)
    db.connect()
    db.create_tables()
    db.load_from_csv(csv_path)

    if verbose:
        # Test queries
        print("\n" + "=" * 60)
        print("Test Queries")
        print("=" * 60)

        print("\nTotal movies:")
        print(db.query("SELECT COUNT(*) as total FROM movies"))

        print("\nHighest rated movies:")
        print(
            db.query("""
            SELECT movie_title, imdb_rating 
            FROM movies 
            WHERE imdb_rating IS NOT NULL
            ORDER BY imdb_rating DESC 
            LIMIT 5
        """)
        )

        print("\nMovies by decade:")
        print(
            db.query("""
            SELECT 
                (release_year / 10) * 10 as decade,
                COUNT(*) as count
            FROM movies
            WHERE release_year IS NOT NULL
            GROUP BY decade
            ORDER BY decade DESC
        """)
        )

    print("\nDatabase ready at datasets/rotten-tomatoes-reviews/prep/movies_meta.db")
    db.close()
