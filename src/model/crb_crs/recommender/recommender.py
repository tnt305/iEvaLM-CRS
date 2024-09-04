"""Recommender component for CRB-CRS model.

This component is responsible for replacing placeholders, if any, in the
retrieved response with appropriate movie information.
"""

from __future__ import annotations

import os
import pickle
from abc import ABC, abstractmethod
from typing import List


class Recommender(ABC):
    def __init__(self) -> None:
        """Initializes the recommender."""
        pass

    @abstractmethod
    def get_recommendations(self, context: List[str]) -> List[str]:
        """Gets recommendations based on the conversation context.

        Args:
            context: Conversation context.

        Raises:
            NotImplementedError: If the method is not implemented in the
              subclass.

        Returns:
            List of recommended items.
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
