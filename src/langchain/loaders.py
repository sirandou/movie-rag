from typing import List, Iterator
from langchain.schema import Document
from langchain.document_loaders.base import BaseLoader
import pandas as pd

from src.data.document_creators import (
    create_plot_docs,
    create_review_docs,
    create_poster_docs,
)


class MovieTextDocumentLoader(BaseLoader):
    """
    Langchain loader that uses custom document creation functions, and converts the documents to LangChain format.
    Supports plots, reviews. DOES NOT support posters.
    """

    def __init__(
        self,
        plots_path: str | None = None,
        reviews_path: str | None = None,
        max_movies: int | None = None,
    ):
        """
        Initialize loader.

        Args:
            plots_path: Path to movie_plots.csv
            reviews_path: Path to reviews_w_movies_full.csv
            max_movies: Limit number of movies
        """
        self.plots_path = plots_path
        self.reviews_path = reviews_path
        self.max_movies = max_movies

    def load(
        self,
        movie_id_cols: list[str] | None = None,
        text_metadata_cols: list[str] | None = None,
        obj_metadata_cols: list[str] | None = None,
    ) -> List[Document]:
        """Load all movie documents in LangChain format.
        Returns:
            List of LangChain Document objects
        """
        # Default text metadata, obj metadata, and id columns
        if obj_metadata_cols is None:
            obj_metadata_cols = [
                "rotten_tomatoes_link",
                "movie_title",
                "release_year",
                "original_release_date",
                "authors",
                "actors",
                "production_company",
                "genres",
                "imdb_rating",
                "box_office",
                "content_rating",
                "runtime",
                "tomatometer_rating",
                "tomatometer_count",
                "audience_rating",
                "audience_count",
                "tomatometer_top_critics_count",
                "tomatometer_fresh_critics_count",
                "tomatometer_rotten_critics_count",
            ]
        if text_metadata_cols is None:
            text_metadata_cols = [
                "movie_title",
                "release_year",
                "directors",
                "genres",
                "content_rating",
                "runtime",
                "tomatometer_rating",
                "box_office",
                "awards",
                "imdb_rating",
                "audience_rating",
                "actors",
            ]
        if movie_id_cols is None:
            movie_id_cols = [
                "rotten_tomatoes_link",
                "movie_title",
                "release_year",
            ]

        if self.max_movies:
            selected_movies_df = self._sample_movies()

        documents = []

        # Load plots
        if self.plots_path:
            documents = documents + self._load_plots(
                text_metadata_cols,
                obj_metadata_cols,
                selected_movies_df if self.max_movies else None,
            )

        # Load reviews
        if self.reviews_path:
            documents = documents + self._load_reviews(
                text_metadata_cols,
                obj_metadata_cols,
                movie_id_cols,
                selected_movies_df if self.max_movies else None,
            )

        print(f"✓ Total: {len(documents)} reviews and plots documents")
        return documents

    def _sample_movies(self) -> pd.DataFrame:
        """Sample a subset of movies of size self.max_movies for quicker testing."""
        print(f"Limiting to {self.max_movies} movies")
        if self.reviews_path:
            df = pd.read_csv(
                self.reviews_path,
                usecols=[
                    "movie_title",
                    "rotten_tomatoes_link",
                    "original_release_date",
                ],
            )
        elif self.plots_path:
            df = pd.read_csv(
                self.reviews_path,
                usecols=[
                    "movie_title",
                    "rotten_tomatoes_link",
                    "original_release_date",
                ],
            )
        else:
            raise ValueError("At least one data path must be provided")

        df["release_year"] = pd.to_datetime(df["original_release_date"]).dt.year
        df = df.drop_duplicates()
        df = df.sample(n=self.max_movies, random_state=42).reset_index(drop=True)
        return df

    def _load_plots(
        self,
        text_metadata_cols: list[str],
        obj_metadata_cols: list[str],
        selected_movies: pd.DataFrame | None,
    ) -> List[Document]:
        """Load only plot documents."""
        documents = []
        print(f"Loading plots from {self.plots_path}...")

        plots_df = pd.read_csv(self.plots_path)
        plots_df["release_year"] = pd.to_datetime(
            plots_df["original_release_date"]
        ).dt.year

        if self.max_movies:
            plots_df = plots_df.merge(
                selected_movies, on=selected_movies.columns.tolist(), how="inner"
            )

        plot_docs = create_plot_docs(plots_df, text_metadata_cols, obj_metadata_cols)

        # Convert to LangChain format
        for doc in plot_docs:
            documents.append(
                Document(page_content=doc["page_content"], metadata=doc["metadata"])
            )

        print(f"  ✓ {len(plot_docs)} plot documents")
        return documents

    def _load_reviews(
        self,
        text_metadata_cols: list[str],
        obj_metadata_cols: list[str],
        movie_id_cols: list[str],
        selected_movies: pd.DataFrame | None,
    ) -> List[Document]:
        """Load only review documents."""
        documents = []
        print(f"Loading reviews from {self.reviews_path}...")

        reviews_df = pd.read_csv(self.reviews_path)
        reviews_df["release_year"] = pd.to_datetime(
            reviews_df["original_release_date"]
        ).dt.year

        if self.max_movies:
            reviews_df = reviews_df.merge(
                selected_movies, on=selected_movies.columns.tolist(), how="inner"
            )

        review_docs = create_review_docs(
            reviews_df, text_metadata_cols, obj_metadata_cols, movie_id_cols
        )

        # Convert to LangChain format
        for doc in review_docs:
            documents.append(
                Document(page_content=doc["page_content"], metadata=doc["metadata"])
            )

        print(f"  ✓ {len(review_docs)} review documents")
        return documents

    def lazy_load(self) -> Iterator[Document]:
        """Lazy loading for large datasets."""
        return iter(self.load())


class MoviePosterDocumentLoader(BaseLoader):
    """
    Loader that uses custom document creation functions for posters. No conversion to langchain needed, as
    even in langchain visual retriever documents are handled as dicts.
    """

    def __init__(
        self,
        posters_path: str,
        max_movies: int | None = None,
    ):
        """
        Initialize loader.

        Args:
            posters_path: Path to movie_posters.csv
            max_movies: Limit number of movies
        """
        self.posters_path = posters_path
        self.max_movies = max_movies

    def load(self, obj_metadata_cols: list[str] | None = None) -> List[dict]:
        """Load all movie poster documents in dict format.
        Returns:
            List of dict objects
        """
        # Default obj metadata
        if obj_metadata_cols is None:
            obj_metadata_cols = [
                "rotten_tomatoes_link",
                "movie_title",
                "release_year",
                "original_release_date",
                "authors",
                "actors",
                "production_company",
                "genres",
                "imdb_rating",
                "box_office",
                "content_rating",
                "runtime",
                "tomatometer_rating",
                "tomatometer_count",
                "audience_rating",
                "audience_count",
                "tomatometer_top_critics_count",
                "tomatometer_fresh_critics_count",
                "tomatometer_rotten_critics_count",
            ]

        # Load posters
        print(f"Loading posters from {self.posters_path}...")

        posters_df = pd.read_csv(self.posters_path)
        posters_df["release_year"] = pd.to_datetime(
            posters_df["original_release_date"]
        ).dt.year

        # Sample movies if needed
        if self.max_movies:
            posters_df = posters_df.drop_duplicates(
                subset=["movie_title", "rotten_tomatoes_link", "original_release_date"]
            )
            posters_df = posters_df.sample(
                n=self.max_movies, random_state=42
            ).reset_index(drop=True)

        posters_docs = create_poster_docs(posters_df, obj_metadata_cols)

        print(f"  ✓ {len(posters_docs)} poster documents")
        return posters_docs

    def lazy_load(self) -> Iterator[dict]:
        """Lazy loading for large datasets."""
        return iter(self.load())
