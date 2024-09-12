"""Movie recommender and metadata integration component for CRB-CRS model.

This component is responsible for replacing placeholders, if any, in the
retrieved response with appropriate movie information.

Adapted from original code:
https://github.com/ahtsham58/CRB-CRS/tree/main

TODO: Improve the code to reduce redundancy and improve efficiency.
"""

from __future__ import annotations

import logging
import os
import pickle
import random
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import linear_kernel

from src.model.crb_crs.recommender.recommender import Recommender
from src.model.crb_crs.retriever.retriever import CRS_PREFIX
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
            os.path.join(self.movielens_data_folder, "movies_metadata.csv")
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
        movies_with_genres["title_formatted"] = movies_with_genres[
            "title"
        ].apply(lambda x: re.sub(r"\(\d{4}\)$", "", x).lower().strip())
        movies_content = movies_with_genres.drop(
            columns=[
                "movieId",
                "rating_mean",
                "title",
                "title_formatted",
                "genres",
                "year",
                "imdbID",
                "directors",
                "actors",
                "movielensID",
                "country",
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

        try:
            title = self.get_movie_title(input_item_id)

            if len(title) < 2:
                return []

            genres = (
                self.movie_mentions_df.loc[[int(input_item_id)]]["genres"]
                .iloc[0]
                .split("|")
            )
            idx = self.movielens_index.values.tolist().index(title)
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
        except ValueError:
            logging.error(
                f"Movie title not found for movie ID {input_item_id}."
            )
            pass
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
        try:
            content_df = self.movie_mentions_df.loc[
                :, ~self.movie_mentions_df.columns.str.contains("^Unnamed")
            ]
            idx = pd.Series(content_df.index, self.movie_mentions_df["title"])
            title = self.get_movie_title(input_item_id)

            if len(title) < 2:
                return []

            movie_index = idx[title]
            similarity_scores = self.cosine_similarity_matrix[
                movie_index
            ].tolist()
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
        except Exception as e:
            logging.error(
                "An error occurred when getting similar items based on "
                f"content:\n{e}"
            )

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

        try:
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
            similar_movies["vote_count"] = similar_movies["vote_count"].astype(
                int
            )
            similar_movies["vote_average"] = similar_movies[
                "vote_average"
            ].astype(int)
            similar_movies["weighted_rating"] = similar_movies.apply(
                lambda x: (
                    (
                        x["vote_count"]
                        / (x["vote_count"] + m)
                        * x["vote_average"]
                    )
                    + (m / (m + x["vote_count"]) * C)
                ),
                axis=1,
            )
            similar_movies = similar_movies.sort_values(
                by="weighted_rating", ascending=False
            ).reset_index()
            similar_movies = similar_movies.sort_values(
                by="year", ascending=False
            )
            similar_movies_titles = similar_movies["title"].values.tolist()
            similar_movies_titles = [
                movie
                for movie in similar_movies_titles
                if movie not in recommended_items
            ]
        except (RuntimeError, TypeError, NameError) as e:
            logging.error(
                "An error occurred when getting similar items based on genre:\n"
                f"{e}"
            )
        return similar_movies_titles[:num_recommendation]

    def detect_previous_item_mentions(
        self, context: List[str], is_user: bool
    ) -> List[str]:
        """Detects items mentioned in the conversation context.

        Args:
            context: Conversation context.
            is_user: Whether the context is from the user or the agent.

        Returns:
            List of item ids corresponding to item mentioned in the
              conversation context.
        """
        mentioned_items = []
        if is_user:
            col = "title_formatted"
        else:
            col = "title"
        for utterance in context:
            for i, movie in enumerate(self.movie_mentions_df[col].values):
                if f" {movie}" in utterance:
                    movie_id = self.movie_mentions_df.index.values[i]
                    mentioned_items.append(movie_id)
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
            user_context, True
        )
        agent_previous_item_mentions = self.detect_previous_item_mentions(
            agent_context, False
        )

        # Get genre preferences per user utterance
        preferences_per_user_utterance = (
            self.get_user_preferences_per_utterance(user_context)
        )

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
        elif len(preferences_per_user_utterance) > 1:
            # Get recommendations based on the last mentioned genre preference
            for preferences in preferences_per_user_utterance[-2::-1]:
                if len(preferences) > 0:
                    genre = preferences[0]
                    break
            recommended_items = self.get_similar_items_genre(
                genre,
                len(agent_previous_item_mentions),
                agent_previous_item_mentions,
            )
        return recommended_items

    def get_user_preferences_per_utterance(
        self, user_context: List[str]
    ) -> List[List[str]]:
        """Gets user preferences per utterance.

        Args:
            user_context: User context (i.e., user utterances in history).

        Returns:
            List of user preferences per utterance.
        """

        preferences_per_user_utterance = [
            list(
                set(utt.split(" ")).intersection(
                    get_preference_keywords("movies")
                )
            )
            for utt in user_context
        ]

        return preferences_per_user_utterance

    def get_movie_title(self, movie_id: str) -> str:
        """Gets the movie title given the movie ID.

        Args:
            movie_id: Movie ID.

        Raises:
            KeyError: If the movie title is not found for the given movie ID.
            Exception: If an error occurs when getting the movie title.

        Returns:
            Movie title.
        """
        try:
            clean_movie_id = re.sub(r"\D", "", movie_id)
            title = self.movie_mentions_df.loc[[int(clean_movie_id)]][
                "title"
            ].iloc[0]
        except KeyError:
            title = ""
            logging.error(f"Movie title not found for movie ID {movie_id}.")
        except Exception as e:
            title = ""
            logging.error(
                f"An error occurred when getting movie title for movie ID "
                f"{movie_id}:\n{e}"
            )
        return title

    def replace_item_ids_with_recommendations(
        self,
        response: str,
        original_item_ids: List[str],
        recommended_items: List[str] = [],
    ) -> str:
        """Replaces item ids in a response with recommended items.

        If no recommended items are available, the item ids are replaced with
        their original titles.

        Args:
            response: Response containing item ids.
            original_item_ids: List of original item ids.
            recommended_items: List of recommended items. Defaults to an empty
              list.

        Returns:
            Response with item ids replaced by recommended items.
        """
        if len(original_item_ids) == len(recommended_items):
            for item_id, recommended_item in zip(
                original_item_ids, recommended_items
            ):
                response = response.replace(f"@{item_id}", recommended_item)
        else:
            # There is a mismatch between the number of item ids and the number
            # of recommended items. In this case, we replace item ids with
            # their original titles.
            for i, item_id in enumerate(original_item_ids):
                try:
                    title = recommended_items[i]
                except IndexError:
                    title = self.get_movie_title(item_id)
                response = response.replace(f"@{item_id}", title)
        return response

    def integrate_domain_metadata(
        self,
        context: List[str],
        response: str,
    ) -> str:
        """Integrates domain metadata into the response.

        In this case, the metadata consists of genre, plot, and actor
        information.

        Args:
            context: Conversation context
            response: Response to integrate domain metadata into.

        Returns:
            Response with domain metadata integrated.
        """
        # Get last movie mentioned by the agent
        agent_context = [utt for utt in context[1::2]]
        items = self.detect_previous_item_mentions(agent_context, False)
        last_movie_mentioned = items[-1] if len(items) > 0 else None

        last_user_utterance = context[-1].lower()

        if last_movie_mentioned is not None:
            movie_metadata = self.movie_mentions_df.loc[
                [int(last_movie_mentioned)]
            ]
            if last_user_utterance.__contains__(
                "who is"
            ) or last_user_utterance.lower().__contains__("who's"):
                # Integrate actor information
                actors = movie_metadata["actors"].iloc[0]
                if len(actors) > 0:
                    return f"{CRS_PREFIX} It stars {actors}."
            if (
                last_user_utterance.__contains__("it about")
                or last_user_utterance.__contains__("plot")
                or last_user_utterance.__contains__("that about")
            ):
                # Integrate plot information
                movie_title = re.sub(
                    r"\(\d{4}\)$", "", movie_metadata["title"].iloc[0]
                ).strip()
                plot = self.movie_metadata_df[
                    self.movie_metadata_df["title"] == movie_title
                ]["overview"].iloc[0]
                if len(plot) > 0:
                    return f"{CRS_PREFIX} {plot}"

        response_preference_tokens = list(
            set(response.split(" ")).intersection(
                get_preference_keywords("movies")
            )
        )
        if len(response_preference_tokens) > 0:
            # Integrate genre information

            # Not optimal before the movie ids are known before in the pipeline
            # TODO: Update implementation to improve efficiency
            response_movie_ids = self.detect_previous_item_mentions(
                [response], False
            )

            user_context = [utt for utt in context[::2]]
            preferences_per_user_utterance = (
                self.get_user_preferences_per_utterance(user_context)
            )
            if (
                len(response_movie_ids) > 0
                and len(preferences_per_user_utterance[-1]) > 0
            ):
                response = self.replace_genre(
                    response, response_preference_tokens, response_movie_ids[0]
                )
            elif len(preferences_per_user_utterance) > 1:
                for preferences in preferences_per_user_utterance[::-1]:
                    if len(preferences) > 0:
                        genre = preferences[0]
                        break
                return response.replace(
                    response_preference_tokens[-1],
                    genre,
                )
        return response

    def replace_genre(
        self, response: str, movie_preference_tokens: List[str], movie_id: str
    ):
        """Replaces genre in the response with the genre of the movie.

        Args:
            response: Response containing genre.
            movie_preference_tokens: List of movie preference tokens.
            movie_id: Movie ID.

        Returns:
            Response with genre replaced by the genre of the movie.
        """
        if (
            response.lower().__contains__("not a")
            and len(movie_preference_tokens) > 0
        ):
            return response

        movie_metadata = self.movie_mentions_df[[int(movie_id)]]
        genres = movie_metadata["genres"]
        if len(genres) > 0:
            genres = genres.iloc[0].split("|")
            for i, preference_token in enumerate(movie_preference_tokens):
                if i > len(genres):
                    response = response.replace(preference_token, "")
                else:
                    response = response.replace(preference_token, genres[i])
                    response = response.replace(
                        preference_token.title(), genres[i]
                    )
                    response = response.replace("comedy", "funny")
                    response = response.replace("romance", "romantic")
                    # Remove redundant genre mentions
                    temp = re.sub(r"\b(\w+)\b\s+\1\b", r"\1", response)
                    unique_words = dict.fromkeys(temp.split())
                    response = " ".join(unique_words)
        else:
            # Genre information is not available for the movie. Use a random
            # adjective as a placeholder.
            response = response.replace(
                movie_preference_tokens[0],
                random.choice([["good", "great", "nice", "awesome", "fine"]]),
            )
            for token in movie_preference_tokens[1:]:
                response = response.replace(token, "")
        return response


if __name__ == "__main__":
    recommender = MovieRecommender(
        "data/models/crb_crs_redial/matrix_factorization"
    )
    context = ["I like comedies."]
    recommendations = recommender.get_recommendations(context)
    print(recommendations)
    recommender.save("data/models/crb_crs_redial/movie_recommender.pkl")
