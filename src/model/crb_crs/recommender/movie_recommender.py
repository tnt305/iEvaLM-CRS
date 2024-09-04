"""Recommender component for CRB-CRS model.

This component is responsible for replacing placeholders, if any, in the
retrieved response with appropriate movie information.

Adapted from original code:
https://github.com/ahtsham58/CRB-CRS/tree/main
"""

from __future__ import annotations

import os
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import linear_kernel

from src.model.crb_crs.recommender.recommender import Recommender
from src.model.crb_crs.utils_preprocessing import get_preference_keywords

DEFAULT_MOVIELENS_DATA_FOLDER = "data/movielens"


class MovieRecommender(Recommender):
    def __init__(
        self,
        matrix_factorization_folder: str,
        movielens_data_folder: str = DEFAULT_MOVIELENS_DATA_FOLDER,
    ) -> None:
        """Initializes the recommender.

        Args:
            matrix_factorization_folder: Path to folder with matrix
              factorization data.
            movielens_data_folder: Path to folder with MovieLens data.
        """
        super().__init__()
        self.movielens_data_folder = movielens_data_folder
        self.movie_metadata_df = pd.read_csv(
            os.path.join(self.movielens_data_folder, "movie_metadata.csv")
        )

        os.makedirs(matrix_factorization_folder, exist_ok=True)
        self.matrix_factorization_folder = matrix_factorization_folder

        model_path = os.path.join(
            self.matrix_factorization_folder, "matrix_factorization.npy"
        )
        index_path = os.path.join(
            self.matrix_factorization_folder, "movielens_index.pkl"
        )
        if os.path.exists(model_path) and os.path.exists(index_path):
            self.matrix_factorization = np.load(model_path)
            self.movielens_index = pickle.load(open(index_path, "rb"))
        else:
            self.initialize_truncated_svd(save=True)

        self._create_cosine_similarity_matrix()

    def _create_cosine_similarity_matrix(self) -> None:
        """Creates the cosine similarity matrix."""
        self.content_matrix, self.movie_mentions_df = (
            self._get_content_matrix()
        )
        self.cosine_similarity_matrix = linear_kernel(
            self.content_matrix, self.content_matrix
        )

    def _get_content_matrix(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Gets the content matrix for movies.

        The data is taken from the MovieLens dataset.

        Returns:
            Tuple of content matrix and DataFrame.
        """
        movie_ratings = pd.read_csv(
            os.path.join(self.movielens_data_folder, "movies_rating_data.csv"),
            encoding="Latin1",
        ).reset_index()
        movies = pd.read_csv(
            os.path.join(self.movielens_data_folder, "movies_data.csv"),
            encoding="Latin1",
        )

        # Add column year to movie ratings
        movie_ratings["year"] = movie_ratings["title"].str.extract(
            r"\((\d{4})\)"
        )

        # Get list of all genres
        genres = set()
        for genre in movies["genres"].str.split("|"):
            genres.update(genre)

        # Create a column for each genre
        movies_with_genres = movie_ratings.copy()
        for genre in genres:
            movies_with_genres[genre] = movies_with_genres[
                "genres"
            ].str.contains(genre)

        movies_with_genres = movies_with_genres.set_index("databaseId")
        movies_content = movies_with_genres.drop(
            columns=[
                "movieId",
                "rating_mean",
                "title",
                "genres",
                "year",
            ]
        )
        movies_content_matrix = movies_content.values
        movies_content_matrix = np.delete(movies_content_matrix, 0, 1)
        movies_content_matrix = np.delete(movies_content_matrix, 0, 1)
        return movies_content_matrix, movies_with_genres

    def initialize_truncated_svd(self, save: bool = False) -> None:
        """Initializes the TruncatedSVD model.

        This model is used for matrix factorization.
        """
        self.user_ratings_df = pd.read_csv(
            "data/movielens/ratings_latest.csv",
            usecols=["userId", "movieId", "rating"],
        )[:1500000]
        self.movie_df = pd.read_csv(
            os.path.join(self.movielens_data_folder, "movies.csv")
        )
        self.movie_df["year"] = self.movie_df["title"].str.extract(
            r"\((\d{4})\)"
        )

        movie_ratings_df = pd.merge(
            self.user_ratings_df, self.movie_df, on="movieId"
        ).dropna(axis=0, subset=["title"])
        user_ratings = (
            movie_ratings_df.groupby(by="title")["rating"]
            .mean()
            .reset_index()
            .rename(columns={"rating": "ratingMean"})[["title", "ratingMean"]]
            .merge(movie_ratings_df, on="title", how="right")
            .drop_duplicates(["userId", "title"])
            .pivot(index="userId", columns="title", values="rating")
            .fillna(0)
        )

        x = user_ratings.values.T

        svd = TruncatedSVD(n_components=20, random_state=42)
        matrix = svd.fit_transform(x)
        self.matrix_factorization = np.corrcoef(matrix)
        self.movielens_index = user_ratings.columns

        if save:
            np.save(
                os.path.join(
                    self.matrix_factorization_folder,
                    "matrix_factorization.npy",
                ),
                self.matrix_factorization,
            )
            with open(
                os.path.join(
                    self.matrix_factorization_folder, "movielens_index.pkl"
                ),
                "wb",
            ) as f:
                pickle.dump(self.movielens_index, f)

    def get_similar_items_ratings(
        self,
        input_item_id: str,
        num_recommendation: int,
        recommended_items: List[str],
    ) -> List[str]:
        """Gets the most similar items based on the ratings.

        Args:
            input_item_id: Input item ID.
            num_recommendation: Number of recommendations to return.
            recommended_items: List of already recommended items.

        Returns:
            List of similar items.
        """
        similar_movies_titles = []
        title = self.get_movie_title(input_item_id)

        if len(title) < 2:
            return []

        genres = (
            self.movie_mentions_df.loc[
                self.movie_mentions_df["databaseId"] == int(input_item_id)
            ]["genres"]
            .iloc[0]
            .split("|")
        )
        idx = self.movielens_index.index(title)
        similarity_scores = list(enumerate(self.matrix_factorization[idx]))
        similarity_scores = sorted(
            similarity_scores, key=lambda x: x[1], reverse=True
        )[1:]
        similar_movies = pd.DataFrame(
            [self.movielens_index[i[0]] for i in similarity_scores],
            columns=["title"],
        )
        similar_movies = similar_movies.merge(
            self.movie_df[["title", "genres", "year", "ratingMean"]],
            how="left",
            on="title",
        )
        similar_movies["matchCount"] = similar_movies["genres"].apply(
            lambda x: len(set(x.split("|")).intersection(genres))
        )
        similar_movies = similar_movies.sort_values(
            by="ratingMean", ascending=False
        ).reset_index()
        similar_movies = similar_movies.sort_values(
            by="year", ascending=False
        ).reset_index()
        similar_movies = similar_movies.sort_values(
            by="matchCount", ascending=False
        )

        similar_movies_titles = similar_movies["title"].values.tolist()
        similar_movies_titles = [
            movie
            for movie in similar_movies_titles
            if movie not in recommended_items
        ]

        return similar_movies_titles[:num_recommendation]

    def get_similar_items_content(
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
        similar_movies_titles = []
        content_df = self.movie_mentions_df.loc[
            :, ~self.movie_mentions_df.columns.str.contains("^Unnamed")
        ]
        idx = pd.Series(content_df.index, self.movie_mentions_df["title"])
        title = self.get_movie_title(input_item_id)

        if len(title) < 2:
            return []

        movie_index = idx[title]
        similarity_scores = self.cosine_similarity_matrix[movie_index].tolist()
        scores = pd.DataFrame(
            similarity_scores, columns=["scores"]
        ).sort_values(by="scores", ascending=False)
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
            movie
            for movie in similar_movies_titles
            if movie not in recommended_items
        ]

        return similar_movies_titles[:num_recommendation]

    def get_similar_items_genre(
        self, genre: str, num_recommendation: int, recommended_items: List[str]
    ) -> List[str]:
        """Gets the most similar items based on the genre.

        Args:
            genre: Genre.
            num_recommendation: Number of recommendations to return.
            recommended_items: List of already recommended items.

        Returns:
            List of similar items.
        """
        similar_movies_titles = []
        # Convert genre to the format used in the dataset
        if genre.lower() == "scary":
            genre = "Horror"
        elif genre.lower() == "romantic" or genre.lower() == "romances":
            genre = "Romance"
        elif genre.lower() == "preference":
            genre = "Adventure"
        elif genre.lower() == "suspense":
            genre = "Thriller"
        elif genre.lower() == "funny":
            genre = "Comedy"
        elif genre.lower() == "comedies":
            genre = "Comedy"
        elif genre.lower() == "scifi":
            genre = "Science Fiction"
        elif genre.lower() == "kids":
            genre = "Comedy"
        elif genre.lower() == "mysteries":
            genre = "mystery"
        genre = genre.title()

        # Get movies with the specified genre
        movies_with_genre = self.movie_metadata_df[
            self.movie_metadata_df["genre"] == genre
        ]

        vote_counts = movies_with_genre[
            movies_with_genre["vote_count"].notnull()
        ]["vote_count"].astype(int)
        vote_averages = movies_with_genre[
            movies_with_genre["vote_average"].notnull()
        ]["vote_average"].astype(int)
        C = vote_averages.mean()
        m = vote_counts.quantile(0.85)

        similar_movies = movies_with_genre[
            (movies_with_genre["vote_count"] >= m)
            & (movies_with_genre["vote_count"].notnull())
            & (movies_with_genre["vote_average"].notnull())
        ][
            [
                "title",
                "year",
                "vote_count",
                "vote_average",
                "popularity",
                "genre",
            ]
        ]
        similar_movies["vote_count"] = similar_movies["vote_count"].astype(int)
        similar_movies["vote_average"] = similar_movies["vote_average"].astype(
            int
        )
        similar_movies["weighted_rating"] = similar_movies.apply(
            lambda x: (
                (x["vote_count"] / (x["vote_count"] + m) * x["vote_average"])
                + (m / (m + x["vote_count"]) * C)
            ),
            axis=1,
        )
        similar_movies = similar_movies.sort_values(
            by="weighted_rating", ascending=False
        ).reset_index()
        similar_movies = similar_movies.sort_values(by="year", ascending=False)
        similar_movies_titles = similar_movies["title"].values.tolist()
        similar_movies_titles = [
            movie
            for movie in similar_movies_titles
            if movie not in recommended_items
        ]
        return similar_movies_titles[:num_recommendation]

    def detect_previous_item_mentions(self, context: List[str]) -> List[str]:
        """Detects items mentioned in the conversation context.

        Args:
            context: Conversation context.

        Returns:
            List of items mentioned in the conversation context.
        """
        mentioned_items = []
        for utterance in context:
            for movie in self.movie_mentions_df["title"].values:
                if movie in utterance:
                    mentioned_items.append(movie)
        return mentioned_items

    def get_recommendations(self, context: List[str]) -> List[str]:
        """Gets recommendations.

        Args:
            context: Conversation context.

        Returns:
            List of recommendations.
        """
        recommended_items = []

        # Split the context into user and agent utterances. Assume that the
        # context starts with a user utterance and alternates between user and
        # agent utterances.
        user_context = [utt for utt in context[::2]]
        agent_context = [utt for utt in context[1::2]]

        # Detect items mentioned in the conversation context for each dialogue
        # participant
        user_previous_item_mentions = self.detect_previous_item_mentions(
            user_context
        )
        agent_previous_item_mentions = self.detect_previous_item_mentions(
            agent_context
        )

        # Get genre preferences based on the last user utterance
        preferences_per_user_utterance = [
            list(
                set(utt.split(" ")).intersection(
                    get_preference_keywords("movies")
                )
            )
            for utt in user_context
        ]

        if len(user_previous_item_mentions) > 0:
            # Get recommendations based on the previous item mentions
            recommended_items = self.get_similar_items_ratings(
                user_previous_item_mentions[-1],
                len(user_previous_item_mentions),
                agent_previous_item_mentions,
            )
            if len(recommended_items) == 0:
                recommended_items = self.get_similar_items_content(
                    user_previous_item_mentions[-1],
                    len(user_previous_item_mentions),
                    agent_previous_item_mentions,
                )
        elif len(preferences_per_user_utterance[-1]) > 0:
            # Get recommendations based on last user utterance preferences
            recommended_items = self.get_similar_items_genre(
                preferences_per_user_utterance[-1][-1],
                len(preferences_per_user_utterance),
                agent_previous_item_mentions,
            )
        else:
            # Get recommendations based on the last mentioned genre preference
            recommended_items = self.get_similar_items_genre(
                preferences_per_user_utterance[-2][-1],
                len(agent_previous_item_mentions),
                agent_previous_item_mentions,
            )
        return recommended_items

    def get_movie_title(self, movie_id: str) -> str:
        """Gets the movie title given the movie ID.

        Args:
            movie_id: Movie ID.

        Returns:
            Movie title.
        """
        return self.movie_mentions_df.loc[
            self.movie_mentions_df["databaseId"] == int(movie_id)
        ]["title"].iloc[0]


if __name__ == "__main__":
    recommender = MovieRecommender(
        "data/models/crb_crs_redial/matrix_factorization"
    )
    context = ["I like romantic movies."]
    recommendations = recommender.get_recommendations(context)
    print(recommendations)
    recommender.save("data/models/crb_crs_redial/movie_recommender.pkl")
