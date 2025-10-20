"""Document creation functions for movie plots and reviews.

This module provides functions to convert DataFrames of movie plots and reviews
into structured dictionary documents suitable for LangChain and other RAG systems.
"""

from typing import List
import pandas as pd


def create_plot_docs(
    plots_df: pd.DataFrame,
    text_metadata_columns: List[str],
    obj_metadata_columns: List[str],
) -> List[dict]:
    """Converts a plots DataFrame into dictionary documents with chosen metadata.

    Args:
        plots_df: DataFrame containing movie plot data
        text_metadata_columns: Columns to prepend to the text content
        obj_metadata_columns: Columns to include in the metadata dict

    Returns:
        List of dictionaries with 'page_content' and 'metadata' keys
    """
    # remove rows with nan plots
    plots_df = plots_df.dropna(subset=["plot"])

    plot_docs = []

    for _, row in plots_df.iterrows():
        # Prepend metadata
        metadata_text = ""
        for col in text_metadata_columns:
            if col in row and pd.notna(row[col]):
                metadata_text += f"{col.replace('_', ' ').capitalize()}: {row[col]}\n"

        # Combine metadata + plot
        full_text = metadata_text + "\n\nPlot: " + str(row["plot"])

        # Create Document
        temp_doc = {"page_content": full_text, "metadata": {"source": "plot"}}
        for col in obj_metadata_columns:
            if col in row and pd.notna(row[col]):
                temp_doc["metadata"][col] = row[col]

        plot_docs.append(temp_doc)

    print(f"Created {len(plot_docs)} plot docs.")
    return plot_docs


def create_review_docs(
    reviews_df: pd.DataFrame,
    text_metadata_columns: List[str],
    obj_metadata_columns: List[str],
    movie_id_cols: List[str],
) -> List[dict]:
    """Converts a reviews DataFrame into dictionary documents with chosen metadata.

    Aggregates multiple reviews per movie into a single document.

    Args:
        reviews_df: DataFrame containing review data
        text_metadata_columns: Columns to prepend to the text content
        obj_metadata_columns: Columns to include in the metadata dict
        movie_id_cols: Columns to group reviews by (identifies unique movies)

    Returns:
        List of dictionaries with 'page_content' and 'metadata' keys
    """
    # remove rows with nan plots
    reviews_df = reviews_df.dropna(subset=["review_content"])

    review_docs = []

    reviews_grouped = reviews_df.groupby(movie_id_cols)

    for movie_id, group in reviews_grouped:
        # Take first row of each movie to get metadata
        first_row = group.iloc[0]
        metadata_text = ""
        for col in text_metadata_columns:
            if col in first_row and pd.notna(first_row[col]):
                metadata_text += (
                    f"{col.replace('_', ' ').capitalize()}: {str(first_row[col])}\n"
                )

        # Combine multiple reviews
        combined_reviews = "\n".join(
            [f"Review: {r}" for r in group["review_content"].tolist()]
        )
        full_text = metadata_text + "\n\nReviews:\n" + combined_reviews

        # Create Document
        temp_doc = {"page_content": full_text, "metadata": {"source": "review"}}
        for col in obj_metadata_columns:
            if col in first_row and pd.notna(first_row[col]):
                temp_doc["metadata"][col] = first_row[col]

        review_docs.append(temp_doc)

    print(f"Created {len(review_docs)} review docs.")
    return review_docs


def create_poster_docs(
    posters_df: pd.DataFrame,
    obj_metadata_columns: List[str],
) -> List[dict]:
    """Converts a posters DataFrame into dictionary documents with chosen metadata.

    Args:
        posters_df: DataFrame containing movie poster path and metadata
        text_metadata_columns: Columns to prepend to the text content
        obj_metadata_columns: Columns to include in the metadata dict

    Returns:
        List of dictionaries with 'poster_path', 'text_content', and 'metadata' keys
    """
    # remove rows with nan poster paths
    posters_df = posters_df.dropna(subset=["poster_path"])

    poster_docs = []

    for _, row in posters_df.iterrows():
        # Build concise text (CLIP limit: 77 tokens ≈ 60 words)
        parts = []

        # Title
        if "movie_title" in row and pd.notna(row["movie_title"]):
            parts.append(row["movie_title"])

        # Year
        if "release_year" in row and pd.notna(row["release_year"]):
            parts.append(f"({row['release_year']})")

        # Genres (important for visual style)
        if "genres" in row and pd.notna(row["genres"]):
            genres = str(row["genres"])[:50]  # Truncate if too long
            parts.append(genres)

        # Director (only first)
        if "directors" in row and pd.notna(row["directors"]):
            directors = str(row["directors"]).split(",")[0]  # Just first director
            parts.append(f"directed by {directors}")

        # Actors (first 5)
        if "actors" in row and pd.notna(row["actors"]):
            actors = (", ").join(str(row["actors"]).split(",")[:5])  # first 5 actors
            parts.append(f"actors: {actors}")

        # Join all parts
        text_content = ". ".join(parts) + "."

        # Safety: Truncate to roughly 60 words (≈75 tokens with safety margin)
        words = text_content.split()
        if len(words) > 60:
            text_content = " ".join(words[:60]) + "..."

        # Create Document
        temp_doc = {
            "poster_path": row["poster_path"],
            "text_content": text_content,
            "metadata": {"source": "poster"},
        }
        for col in obj_metadata_columns:
            if col in row and pd.notna(row[col]):
                temp_doc["metadata"][col] = row[col]

        poster_docs.append(temp_doc)

    print(f"Created {len(poster_docs)} poster docs.")
    return poster_docs


if __name__ == "__main__":
    from pathlib import Path
    import argparse

    # Get path from user
    parser = argparse.ArgumentParser(
        description="Create document dictionaries from movie plots, reviews, and posters data"
    )
    parser.add_argument(
        "data_path",
        type=Path,
        help="Path to the prep directory containing movie_plots.csv and reviews_w_movies_full.csv",
    )
    args = parser.parse_args()

    path = args.data_path

    # Load data
    plots_df = pd.read_csv(path / "movie_plots.csv")
    reviews_df = pd.read_csv(path / "reviews_w_movies_full.csv")
    posters_df = pd.read_csv(path / "movie_posters.csv")

    # Add year column for date
    plots_df["release_year"] = pd.to_datetime(plots_df["original_release_date"]).dt.year
    reviews_df["release_year"] = pd.to_datetime(
        reviews_df["original_release_date"]
    ).dt.year
    posters_df["release_year"] = pd.to_datetime(
        posters_df["original_release_date"]
    ).dt.year

    # Define metadata columns
    movie_id_cols = ["rotten_tomatoes_link", "movie_title", "release_year"]
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

    # Create documents
    plot_docs = create_plot_docs(plots_df, text_metadata_cols, obj_metadata_cols)
    review_docs = create_review_docs(
        reviews_df, text_metadata_cols, obj_metadata_cols, movie_id_cols
    )
    poster_docs = create_poster_docs(posters_df, obj_metadata_cols)

    all_docs = plot_docs + review_docs + poster_docs
    print(f"\nTotal documents: {len(all_docs)}")

    # View sample documents
    print("\nOne plot doc:")
    print(plot_docs[0]["metadata"])
    print("\n")
    print(plot_docs[0]["page_content"])
    print("\n\n")
    print("One review doc:")
    print(review_docs[0]["metadata"])
    print("\n")
    print(review_docs[0]["page_content"])
    print("\n\n")
    print("One poster doc:")
    print(poster_docs[0]["metadata"])
    print("\n")
    print(poster_docs[0]["poster_path"])
    print("\n")
    print(poster_docs[0]["text_content"])
