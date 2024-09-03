"""Recommender component for CRB-CRS model.

This component is responsible for replacing placeholders, if any, in the
retrieved response with appropriate movie information.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from src.model.crb_crs.recommender.recommender import Recommender


class MovieRecommender(Recommender):
    def __init__(self) -> None:
        """Initializes the recommender."""
        super().__init__()

    def _get_content_matrix(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Gets the content matrix for movies.

        The data is taken from the MovieLens dataset.

        Returns:
            Tuple of content matrix and DataFrame.
        """
        movie_ratings = pd.read_csv(
            "data/movielens/movies_rating_data.csv", encoding="Latin1"
        ).reset_index()
        movies = pd.read_csv("data/movielens/movies_data.csv", encoding="Latin1")

        # Add column year to movie ratings
        movie_ratings["year"] = movie_ratings["title"].str.extract(r"\((\d{4})\)")

        # Get list of all genres
        genres = set()
        for genre in movies["genres"].str.split("|"):
            genres.update(genre)

        # Create a column for each genre
        movies_with_genres = movie_ratings.copy()
        for genre in genres:
            movies_with_genres[genre] = movies_with_genres["genres"].str.contains(genre)
        movies_with_genres = movies_with_genres.set_index("databaseId")
        movies_content = movies_with_genres.drop(
            columns=[
                "movieId",
                "rating_mean",
                "title",
                "genres",
                "year",
                "databaseId",
            ]
        )
        movies_content_matrix = movies_content.values
        movies_content_matrix = np.delete(movies_content_matrix, 0, 1)
        movies_content_matrix = np.delete(movies_content_matrix, 0, 1)
        return movies_content_matrix, movies_content

    def get_similar_items(
        self,
        input_item_id: str,
        num_recommendation: int,
        recommended_items: List[str],
    ) -> List[str]:
        """Gets the most similar items based on the content.

        Args:
            input_item_id: Input item ID.
            num_recommendation: Number of recommendations to return.
            recommended_items: List of already recommended items.

        Returns:
            List of similar items.
        """
        similar_items = []
        content_df = self.content_df.loc[
            :, ~self.content_df.columns.str.contains("^Unnamed")
        ]
        idx = pd.Series(content_df.index, self.content_df["title"])
        title = self.content_df.loc[
            self.content_df["databaseId"] == int(input_item_id)
        ]["title"].iloc[0]

        if len(title) < 2:
            return []

        movie_index = idx[title]
        similarity_scores = self.cosine_similarity_matrix[movie_index].tolist()
        scores = pd.DataFrame(similarity_scores, columns=["scores"]).sort_values(
            by="scores", ascending=False
        )
        similar_movies_indices = scores.index.values.tolist()
        similar_movies = pd.DataFrame(
            content_df[["title", "genres", "year", "rating_mean"]].iloc[
                similar_movies_indices
            ]
        )
        similar_movies = similar_movies[similar_movies["title"] != title]
        similar_movies = similar_movies.sort_values(
            by="rating_mean", ascending=False
        ).reset_index()
        similar_movies = similar_movies.sort_values(
            by="year", ascending=False
        ).reset_index()
        similar_movies_titles = similar_movies["title"].values.tolist()
        similar_movies_titles = [
            movie for movie in similar_movies_titles if movie not in recommended_items
        ]

        return similar_items[:num_recommendation]
