"""Contextual Retrieval-based CRS.

This CRS comprises two main components: a retriever and a recommender. The 
retriever is responsible for retrieving and ranking a set of responses from a
pre-defined corpus. The recommender is responsible for recommending items
whenever the retrieved responses contains an item placeholder.

Adapted from original code:
https://github.com/ahtsham58/CRB-CRS/tree/main
"""

import logging
import os
import re
from typing import Any, Dict, List, Tuple

import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

from src.model.crb_crs.recommender import *
from src.model.crb_crs.retriever.mle_model import NGramMLE
from src.model.crb_crs.retriever.retriever import Retriever


class CRBCRSModel:
    def __init__(
        self,
        dataset: str,
        domain: str,
        corpus_folder: str,
        mle_model_path: str,
        recommender_path: str,
    ) -> None:
        """Initializes the CRB-CRS model.

        Args:
            dataset: Dataset name.
            domain: Domain of application.
            corpus_folder: Path to the folder containing the corpus.
            mle_model_path: Path to the MLE model.
            recommender_path: Path to the recommender model.

        Raises:
            FileNotFoundError: If MLE model path does not exist.
        """
        if not os.path.exists(mle_model_path):
            raise FileNotFoundError(
                f"MLE model path {mle_model_path} does not exist."
            )
        mle_model = NGramMLE.load(mle_model_path)
        self.kg_dataset = dataset  # No relation with a KG, naming is kept for compatibility. # noqa

        self.retriever = Retriever(corpus_folder, mle_model, dataset, domain)
        self.recommender = Recommender.load(recommender_path)

    def get_rec(self, conv_dict: Dict[str, Any]):
        """Generates recommendations given a conversation context."""
        pass

    def get_conv(self, conv_dict: Dict[str, Any]):
        """Generates utterance given a conversation context."""
        pass

    def get_response(
        self,
        conv_dict: Dict[str, Any],
        id2entity: Dict[int, str] = None,
        options: Tuple[str, Dict[str, str]] = None,
        state: List[float] = None,
    ) -> Tuple[str, List[float]]:
        """Generates a response given a conversation context.

        Args:
            conv_dict: Conversation context.
            id2entity (not used): Mapping from entity id to entity name.
              Defaults to None.
            options (not used): Prompt with options and dictionary of options.
              Defaults to None.
            state (not used): State of the option choices. Defaults to None.
        Returns:
            Generated response.
        """
        # Retrieval
        context = conv_dict["context"]
        last_user_utterance = context[-1]
        last_user_utterance_tokens = word_tokenize(last_user_utterance)
        candidate_responses = []

        # Get candidates based on the last user utterance
        candidate_responses.extend(self._get_candidates([last_user_utterance]))

        if len(context) > 1:
            # Get candidates based on last user utterance and the previous
            # agent utterance
            candidate_responses.extend(self._get_candidates(context[-2:]))

        if len(context) > 2:
            # Get candidates based on the last user utterance, the previous
            # agent utterance, and the user utterance before that
            candidate_responses.extend(self._get_candidates(context[-3:]))

        if len(context) > 3:
            # Get candidates based on the entire conversation context
            candidate_responses.extend(self._get_candidates(context))

        ranked_candidates = self.retriever.rank_candidates(
            last_user_utterance_tokens, candidate_responses
        )
        retrieved_response = ranked_candidates[0]

        # Recommendation
        recommended_items = []
        original_item_ids = self.get_item_ids_from_retrieved_response(
            retrieved_response
        )
        if original_item_ids:
            recommended_items = self.recommender.get_recommendations(context)

        # Replace item ids in the retrieved response with recommendations
        response = self.recommender.replace_item_ids_with_recommendations(
            retrieved_response, original_item_ids, recommended_items
        )

        # Integrate metadata into the retrieved response
        try:
            response = self.recommender.integrate_domain_metadata(
                context, response
            )
        except Exception as e:
            logging.error(f"Error while integrating metadata: {e}")

        response = self.retriever.remove_utterance_prefix(response)
        return response, None

    def get_item_ids_from_retrieved_response(self, response: str) -> List[str]:
        """Extracts item ids from a retrieved response.

        Args:
            response: Retrieved response.

        Returns:
            List of item ids.
        """
        if self.kg_dataset == "redial":
            return [
                re.sub(r"[@?]", "", id) for id in re.findall(r"@\S+", response)
            ]
        return []

    def _get_candidates(self, context: List[str]) -> List[str]:
        """Gets candidate responses based on the conversation context."""
        input_query = self.retriever.build_query(context)
        candidates = self.retriever.retrieve_candidates(context=input_query)
        return self.retriever.filter_outliers_from_candidates(candidates)

    def get_choice(self, gen_inputs, option, state, conv_dict=None):
        """Generates a choice between options given a conversation context.

        This method is not implemented in this class because the recommendation
        stage is already conditioned on the conversation context."""
        pass
