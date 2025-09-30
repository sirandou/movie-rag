"""Prepare Rotten Tomatoes dataset by cleaning and merging movie and review data."""

import argparse
import os
from pathlib import Path

import pandas as pd


def prep_rotten_tomatoes_data(path: Path) -> pd.DataFrame:
    """Read rotten tomatoes dataset, clean, join them, and save full and a sample version.

    Args:
        path: The project folder path

    Returns:
        DataFrame with merged reviews and movie data
    """
    movies_df = pd.read_csv(
        path / "datasets/rotten-tomatoes-reviews/raw/rotten_tomatoes_movies.csv"
    )
    reviews_df = pd.read_csv(
        path / "datasets/rotten-tomatoes-reviews/raw/rotten_tomatoes_critic_reviews.csv"
    )

    # drop duplicates
    movies_df = movies_df.drop_duplicates()
    reviews_df = reviews_df.drop_duplicates()

    # Remove reviews without content and movies without descriptions
    reviews_df = reviews_df[reviews_df["review_content"].notna()]
    movies_df = movies_df[movies_df["movie_info"].notna()]

    # print stats
    print(f"Dataset Stats:")
    print(f"Total movies: {len(movies_df):,}")
    print(f"Total reviews: {len(reviews_df):,}")
    print(f"Avg reviews per movie: {len(reviews_df) / len(movies_df):.1f}")

    print(f"\nMovie columns: {movies_df.columns.tolist()}")
    print(f"Review columns: {reviews_df.columns.tolist()}")

    print("\nSample review:")
    print(reviews_df.sample(n=1).iloc[0])

    print("\nSample movies:")
    print(movies_df.sample(n=1).iloc[0])

    # merge tables and sort by movie
    reviews_with_movies = reviews_df.merge(
        movies_df, on="rotten_tomatoes_link", how="left"
    )
    reviews_with_movies = reviews_with_movies.sort_values(
        by=["rotten_tomatoes_link"]
    ).reset_index(drop=True)

    # Save full and sample versions
    os.makedirs(path / "datasets/rotten-tomatoes-reviews/prep", exist_ok=True)
    reviews_with_movies.to_csv(
        path / "datasets/rotten-tomatoes-reviews/prep/reviews_w_movies_full.csv",
        index=False,
    )

    # Create sample for development (first 10K reviews)
    sample = reviews_with_movies.head(10000)
    sample.to_csv(
        path / "datasets/rotten-tomatoes-reviews/prep/reviews_w_movies_sample.csv",
        index=False,
    )

    print(f"\nSaved {len(reviews_with_movies):,} reviews to processed/")
    print(f"Created sample with 10,000 reviews for testing")

    return reviews_with_movies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare Rotten Tomatoes dataset by cleaning and merging data"
    )
    parser.add_argument(
        "--path",
        type=Path,
        help="Project folder path",
    )
    args = parser.parse_args()

    reviews_with_movies = prep_rotten_tomatoes_data(args.path)
