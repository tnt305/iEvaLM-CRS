"""Retrieval component for CRB-CRS model.

This component is responsible for retrieving the most relevant utterance from
a set of pre-defined responses given a user query and a conversation history.
"""

import itertools
import math
import os
import re
from typing import List

from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from scipy import spatial
from sent2vec.vectorizer import Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.model.crb_crs.retriever.mle_model import NGramMLE


class Retriever:
    def __init__(
        self, corpus_folder: str, mle_model: NGramMLE, dataset: str
    ) -> None:
        """Initializes the retriever.

        Args:
            corpus_folder: Path to the folder containing the corpus files.
            mle_model: Maximum Likelihood Estimation (MLE) model.
            dataset: Dataset name.

        Raises:
            FileNotFoundError: If the corpus folder is not found.
        """
        if not os.path.exists(corpus_folder):
            raise FileNotFoundError(
                f"Corpus folder not found: {corpus_folder}"
            )

        self.corpus_folder = corpus_folder
        self._create_vectorizers_and_vocabs()
        self.mle_model = mle_model
        self.dataset = dataset

    def _load_original_corpus(self):
        """Loads the original corpus.

        Raises:
            FileNotFoundError: If the original corpus file is not found.
        """
        with open(
            os.path.join(self.corpus_folder, "original_corpus.txt"), "r"
        ) as f:
            self.original_corpus = f.read().splitlines()

    def _load_preprocessed_corpora(self):
        """Loads the preprocessed corpora (w/o stopwords).

        Raises:
            FileNotFoundError: If a preprocessed corpus file is not found.
        """
        with open(
            os.path.join(self.corpus_folder, "preprocessed_corpus.txt"), "r"
        ) as f:
            self.preprocessed_corpus = f.read().splitlines()

        with open(
            os.path.join(
                self.corpus_folder, "preprocessed_corpus_no_stopwords.txt"
            ),
            "r",
        ) as f:
            self.preprocessed_corpus_no_stopwords = f.read().splitlines()

    def _create_vectorizers_and_vocabs(self) -> None:
        """Creates vectorizers for the retriever and builds the vocabularies.

        The vectorizers are based on TF-IDF. Two vectorizers are created: one
        with stopwords and one without stopwords.
        """
        # Load the original and preprocessed corpora
        self._load_original_corpus()
        self._load_preprocessed_corpora()

        self.vectorizer = TfidfVectorizer()
        self.corpus_vocab = self.vectorizer.fit_transform(
            self.preprocessed_corpus
        )
        self.vectorizer_no_stopwords = TfidfVectorizer()
        self.corpus_no_stopwords_vocab = (
            self.vectorizer_no_stopwords.fit_transform(
                self.preprocessed_corpus_no_stopwords
            )
        )

    def retrieve_candidates(
        self, context: str, num_candidates: int = 5
    ) -> List[str]:
        """Retrieves the most relevant candidates given a context.

        Args:
            context: Conversational context.
            num_candidates: Number of candidates to retrieve. Defaults to 5.

        Returns:
            List of retrieved candidates.
        """
        candidates = []
        if len(word_tokenize(context)) > 2:
            context_vector = self.vectorizer_no_stopwords.transform([context])
            cosine_matrix = cosine_similarity(
                context_vector, self.corpus_no_stopwords_vocab
            ).flatten()
        else:
            context_vector = self.vectorizer.transform([context])
            cosine_matrix = cosine_similarity(
                context_vector, self.corpus_vocab
            ).flatten()

        similar_utterances_indices = cosine_matrix.argsort()[:-100:-1]

        for idx in similar_utterances_indices:
            if idx < len(self.original_corpus) - 1:
                try:
                    retrieved_utterance = self.original_corpus[idx + 1]
                    retrieved_utterance = re.sub(
                        r"[^A-Za-z0-9~]+", " ", retrieved_utterance
                    )
                    retrieved_utterance = retrieved_utterance.strip()
                except IndexError:
                    continue
            else:
                retrieved_utterance = self.original_corpus[idx]

            len_retrieved_utterance = word_tokenize(
                retrieved_utterance.split("~")[-1].strip()
            )

            if (
                not retrieved_utterance.__contains__("RECOMMENDER~")
                or len_retrieved_utterance <= 3
                or len_retrieved_utterance > 20
            ):
                continue
            elif not self.original_corpus[idx].__contains__("USER~"):
                continue
            else:
                candidates.append(self.original_corpus[idx + 1])
                if len(candidates) == num_candidates:
                    break

        return candidates

    def build_query(self, context: List[str]) -> str:
        """Builds a query from the context.

        Args:
            context: List of strings representing the utterances.

        Returns:
            Query string.
        """
        return ",".join(context)

    def filter_outliers_from_candidates(
        self, candidates: List[str], num_candidates: int = 5
    ) -> List[str]:
        """Filters out outliers from the list of candidates.

        The outliers are discarded based on mutual similarity score computed
        using BERT model. The logic is to create pairwise combinations of
        candidates, compute the similarity score using BERT embeddings, and then
        keep only valid candidates.

        Args:
            candidates: List of candidates.
            num_candidates: Number of candidates to retrieve. Defaults to 5.

        Raises:
            ValueError: If the list of candidates is empty.

        Returns:
            Filtered list of candidates.
        """
        if not candidates:
            raise ValueError("The list of candidates is empty.")

        candidate_pairs = list(itertools.combinations(candidates, 2))
        num_valid_candidates = math.floor(
            len(candidate_pairs) / num_candidates
        )
        candidate_pairs = list(map(list, candidate_pairs))
        for i, (cand1, cand2) in enumerate(candidate_pairs):
            processed_cand1 = None
            processed_cand2 = None

            vectorizer = Vectorizer()
            vectorizer.bert([processed_cand1, processed_cand2])
            vectors = vectorizer.vectors
            distance = spatial.distance.cosine(vectors[0], vectors[1])
            candidate_pairs[i].append(round(distance, 4))

        # Sort the candidate pairs based on the similarity score
        candidate_pairs.sort(key=lambda x: x[-1], reverse=True)
        filtered_candidates = [
            candidate[0]
            for candidate in candidate_pairs[:num_valid_candidates]
        ]

        return filtered_candidates

    def _item_context(self) -> List[str]:
        """Returns a list of words related to the item context.

        Raises:
            ValueError: If the dataset is not supported.
        """
        if self.dataset == "redial":
            return ["movie", "movies", "movieid"]
        elif self.dataset == "opendialkg":
            return ["movie", "movies", "book", "books", "itemid"]
        raise ValueError(f"Dataset not supported: {self.dataset}")

    def rank_candidates(
        self, user_utterance_tokens: List[str], candidates: List[str]
    ) -> List[str]:
        """Ranks the candidates based on fluency score.

        The fluency score is computed with n-Gram (1-5) Maximum Likelihood
        Probabilistic Language Model.

        Args:
            user_utterance_tokens: List of tokens from the user utterance.
            candidates: List of candidates.

        Returns:
            Ranked list of candidates.
        """
        ranked_candidates = []

        for candidate in candidates:
            processed_candidate = None
            candidate_tokens = word_tokenize(processed_candidate)
            bigrams = list(ngrams(candidate_tokens, 2))
            probability = self.mle_model.utterance_probability(
                processed_candidate, n=2
            )
            avg_score = probability / len(bigrams)
            avg_score = self._update_candidate_rank_score(
                avg_score, user_utterance_tokens, candidate_tokens
            )
            ranked_candidates.append((candidate, avg_score))

        ranked_candidates.sort(key=lambda x: x[1], reverse=True)
        return ranked_candidates

    def _update_candidate_rank_score(
        self,
        avg_score: float,
        user_utterance_tokens: List[str],
        candidate_tokens: List[str],
    ):
        """Updates the candidate rank score.

        The score is updated based on the presence of item context tokens,
        preference keywords, and chit-chat context tokens.

        Args:
            avg_score: Average score from MLE.
            user_utterance_tokens: List of tokens from the user utterance.
            candidate_tokens: List of tokens from the candidate utterance.

        Returns:
            Updated average score.
        """
        chit_chat_context = ["thanks", "bye", "goodbye", "thank"]

        common_tokens_user_utterance = list(
            set(user_utterance_tokens).intersection(self._item_context())
        )
        common_tokens_candidate = list(
            set(candidate_tokens).intersection(self._item_context())
        )
        common_preference_tokens_user_utterance = list(
            set(user_utterance_tokens).intersection(
                self.get_preference_keywords()
            )
        )
        common_preference_tokens_candidate = list(
            set(candidate_tokens).intersection(self.get_preference_keywords())
        )
        common_chit_chat_tokens_user_utterance = list(
            set(chit_chat_context).intersection(user_utterance_tokens)
        )

        if len(common_chit_chat_tokens_user_utterance) > 0:
            avg_score = avg_score + 2.0
        else:
            if (
                len(common_tokens_user_utterance) > 0
                and len(common_tokens_candidate) > 0
            ):
                # Item context tokens are present in both user and candidate
                # utterances
                avg_score = avg_score + 1.0
            if (
                len(common_tokens_user_utterance)
                == len(common_tokens_candidate)
                == 0
            ):
                # No item context tokens in both user and candidate utterances
                avg_score = avg_score + 1.0

            if (
                len(common_preference_tokens_candidate) > 0
                and len(common_preference_tokens_user_utterance) > 0
            ):
                # User and candidate utterances have common preference keywords
                avg_score = avg_score + 5.0

        return avg_score
