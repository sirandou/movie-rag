import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from langchain.tools import tool
import json
import numpy as np
from rapidfuzz import process


class CollaborativeFilteringTool:

    def __init__(
            self,
            reviews_path: str = "/Users/saghar/Desktop/movie-rag/datasets/rotten-tomatoes-reviews/prep/reviews_w_movies_full.csv",
    ):
        """
        Collaborative filtering using cosine similarity.
        Args:
            reviews_path: Path to reviews CSV
        """
        # Load CSV once
        self.df = self._load_and_preprocess_reviews(reviews_path)

        # Build user-movie rating matrix
        # Use movie id as movie titles have overlaps
        self.rating_matrix = self.df.pivot_table(
            index="critic_name", columns="rotten_tomatoes_link", values="rating_normalized", aggfunc="mean"
        )
        print(
            f"Loaded {len(self.rating_matrix)} critics, {len(self.rating_matrix.columns)} movies, {self.rating_matrix.count().sum()} ratings."
        )
        # Fill NaN with 0, only used for cosine similarity computation
        self.rating_matrix_filled = self.rating_matrix.fillna(0)

        # Mappings (handle duplicates)
        df_unique = self.df.drop_duplicates(subset=['rotten_tomatoes_link'])
        # 1. movie_key → imdb_id (for user input). E.g., "Inception (2010)" → "tt1375666"
        self.key_to_id = df_unique.set_index('movie_key')['rotten_tomatoes_link'].to_dict()
        # 2. imdb_id → movie_key (for display). E.g., "tt1375666" → "Inception (2010)"
        self.id_to_key = df_unique.set_index('rotten_tomatoes_link')['movie_key'].to_dict()
        # 3. Also keep title-only lookup for fuzzy matching. E.g., "Inception" → ["Inception (2010)", "Inception (2015)"]
        self.title_to_keys = df_unique.groupby('movie_title')['movie_key'].apply(list).to_dict()

        # lower-case names
        self.key_to_id = {k.lower(): v for k, v in self.key_to_id.items()}
        self.title_to_keys = {k.lower(): [t.lower() for t in v] for k, v in self.title_to_keys.items()}

        # Pre-compute movie-movie similarity matrix (item-based CF)
        self.movie_similarity = cosine_similarity(self.rating_matrix_filled.T)
        self.movie_similarity_df = pd.DataFrame(
            self.movie_similarity,
            index=self.rating_matrix.columns,
            columns=self.rating_matrix.columns,
        )

        print("Computed movie similarity matrix.")

    def _load_and_preprocess_reviews(self, reviews_csv: str) -> pd.DataFrame:
        """Load and preprocess reviews CSV."""
        df = pd.read_csv(reviews_csv)
        # Format and keep numerical ratings only
        df['rating_normalized'] = df['review_score'].apply(normalize_rating)
        df = df.dropna(subset=['rating_normalized'])
        # Drop nan reviewer names
        df = df.dropna(subset=['critic_name'])
        # Add release year
        df["release_year"] = pd.to_datetime(df["original_release_date"]).dt.year
        # Add movie key since titles can overlap
        df['movie_key'] = (
                df['movie_title'] + ' (' +
                df['release_year'].astype(str) + ')'
        )

        return df

    def get_tool(self):
        rating_matrix = self.rating_matrix
        movie_similarity_df = self.movie_similarity_df

        @tool
        def recommend_by_similar_taste(
                movie_titles: str, num_recommendations: int = 5
        ) -> str:
            """
            Recommend movies with similar rating patterns.

            This function uses **item-based collaborative filtering** to find movies
            that share similar critic rating patterns with the given titles.
            It’s best used when you want recommendations based on movies the user liked.

            Typical user prompts:
            - "I loved [X, Y] — what else might I like?"
            - "Find movies similar to [X] based on ratings"
            - "What movies have similar critic rating patterns to [X]?"

            Args:
                movie_titles (str): Comma-separated list of movie titles to use as the seed.
                num_recommendations (int, optional): Number of recommendations to return. Defaults to 5.

            Returns:
                str: JSON string containing recommended movies and their similarity scores.
            """
            try:
                result = {"warnings": ""}

                # Parse input
                titles = [t.strip() for t in movie_titles.split(",")]

                # Find inputs available in dataset
                available_ids = []
                all_keys = list(self.key_to_id.keys())

                for title in titles:
                    # Try exact match with year: "Inception (2010)"
                    if title in self.key_to_id:
                        available_ids.append(self.key_to_id[title])
                        continue

                    # Try title-only lookup -> may have multiple matches
                    if title in self.title_to_keys:
                        matched_keys = self.title_to_keys[title]

                        if len(matched_keys) == 1:
                            # Only one movie with this title
                            available_ids.append(self.key_to_id[matched_keys[0]])
                        else:
                            # Multiple matches: Return error asking for clarification
                            result[
                                "warnings"] += f"Warning: Multiple movies found for {title}. Options: {matched_keys}. "
                        continue

                    # Try Fuzzy match: partial or misspelled titles
                    matches = process.extract(title, all_keys, limit=3, score_cutoff=65)

                    if len(matches) == 1 or (matches[0][1] - matches[1][1] > 10):
                        # Only one strong match or clear top match
                        best_match = matches[0][0]
                        available_ids.append(self.key_to_id[best_match])
                        continue

                    options = [m[0] for m in matches]
                    result[
                        "warnings"] += f"Warning: Multiple close matches found for {title}, it might also not be available in dataset. Options: {options}. "

                if len(available_ids) < 1:
                    return json.dumps({
                        "warnings": result["warnings"],
                        "error": "None of the provided movies were found.",
                    })

                # For each input movie, get similarity scores with other movies. Then
                # aggregate across all input movies by averaging.
                # So a good recommended movie is one that is similar to all input movies.
                similarity_scores = pd.Series(dtype=float)

                for id in available_ids:
                    scores_id = movie_similarity_df[id]
                    if similarity_scores.empty:
                        similarity_scores = scores_id
                    else:
                        similarity_scores = similarity_scores.add(
                            scores_id, fill_value=0
                        )
                similarity_scores = similarity_scores / len(available_ids)

                # Remove input movies
                similarity_scores = similarity_scores.drop(
                    available_ids, errors="ignore"
                )

                # Filter movies with similarity > 0.1 and rated by 5+ critics
                min_critics = 5
                movie_counts = rating_matrix.notna().sum(axis=0)
                valid_movies = movie_counts[movie_counts >= min_critics].index

                similarity_scores = similarity_scores[
                    similarity_scores.index.isin(valid_movies)
                ]
                similarity_scores = similarity_scores[similarity_scores > 0.1]

                # Sort by similarity
                recommendations = similarity_scores.sort_values(ascending=False).head(
                    num_recommendations
                )

                # Get average ratings
                avg_ratings = rating_matrix[recommendations.index].mean(axis=0)

                # Format results
                result = {
                    **result,
                    "input_movies": [self.id_to_key[mid] for mid in available_ids],
                    "method": "Item-based collaborative filtering (cosine similarity)",
                    "recommendations": [
                        {
                            "title (year)": self.id_to_key[movie],
                            "similarity_score": round(float(recommendations[movie]), 3),
                            "avg_critic_rating": round(float(avg_ratings[movie]), 2),
                            "num_critics": int(movie_counts[movie]),
                            "reason": f"Critics rated this similarly to {', '.join([self.id_to_key[mid] for mid in available_ids])}",
                        }
                        for movie in recommendations.index
                    ],
                }

                return json.dumps(result, indent=2)

            except Exception as e:
                return json.dumps({"error": str(e)})

        return recommend_by_similar_taste


def normalize_rating(rating_str: str) -> float:
    """
    Convert various rating formats to 0-10 scale:
    - Fractions: "2.5/4", "3/5"
    - Letter grades: "A", "B+", "C-"
    - Nulls: "<null>", NaN
    """
    if pd.isna(rating_str) or rating_str == '<null>' or rating_str == '':
        return np.nan

    rating_str = str(rating_str).strip().replace(" ", "")

    # Handle fractions (2.5/4, 3/5)
    if '/' in rating_str:
        try:
            score, max_score = rating_str.split('/')
            score = float(score)
            max_score = float(max_score)
            return (score / max_score) * 10
        except:
            return np.nan

    # Handle letter grades (A, B+, C-)
    letter_grade_map = {
        'A+': 10.0, 'A': 9.5, 'A-': 9.0,
        'B+': 8.5, 'B': 8.0, 'B-': 7.5,
        'C+': 7.0, 'C': 6.5, 'C-': 6.0,
        'D+': 5.5, 'D': 5.0, 'D-': 4.5,
        'F': 3.0
    }

    rating_upper = rating_str.upper()
    if rating_upper in letter_grade_map:
        return letter_grade_map[rating_upper]

    # Handle numeric (already 0-10 scale)
    try:
        numeric = float(rating_str)
        # If it's already 0-10, keep it
        if 0 <= numeric <= 10:
            return numeric
        # If it's 0-100, scale down
        elif 0 <= numeric <= 100:
            return numeric / 10
    except:
        pass

    # Unknown format (this never happens with the current data)
    return np.nan
