"""Recommender component for CRB-CRS model.

This component is responsible for replacing placeholders, if any, in the
retrieved response with appropriate movie information.
"""

from __future__ import annotations

import os
import pickle
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel


class Recommender(ABC):
    def __init__(self) -> None:
        """Initializes the recommender."""
        self.content_matrix, self.content_df = self._get_content_matrix()
        self._create_cosine_similarity_matrix()

    def _create_cosine_similarity_matrix(self) -> None:
        """Creates the cosine similarity matrix."""
        self.cosine_similarity_matrix = linear_kernel(
            self.content_matrix, self.content_matrix
        )

    @abstractmethod
    def _get_content_matrix(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Gets the content matrix for items.

        Raises:
            NotImplementedError: If the method is not implemented in the
              subclass.

        Returns:
            Tuple of content matrix and DataFrame.
        """
        raise NotImplementedError

    @abstractmethod
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

        Raises:
            NotImplementedError: If the method is not implemented in the
              subclass.

        Returns:
            List of similar items.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> Recommender:
        """Loads the recommender from the given path.

        Args:
            path: Path to the recommender.

        Raises:
            FileNotFoundError: If the recommender file is not found.

        Returns:
            Loaded recommender.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Recommender file not found: {path}")

        return pickle.load(open(path, "rb"))

    def save(self, path: str) -> None:
        """Saves the recommender to the given path.

        Args:
            path: Path to save the recommender.
        """
        pickle.dump(self, open(path, "wb"))
