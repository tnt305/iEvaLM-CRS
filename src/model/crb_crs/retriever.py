"""Retrieval component for CRB-CRS model.

This component is responsible for retrieving the most relevant utterance from
a set of pre-defined responses given a user query and a conversation history.
"""

import os
import re
from typing import List

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Retriever:
    def __init__(self, corpus_folder: str) -> None:
        """Initializes the retriever.

        Args:
            corpus_folder: Path to the folder containing the corpus files.

        Raises:
            FileNotFoundError: If the corpus folder is not found.
        """
        if not os.path.exists(corpus_folder):
            raise FileNotFoundError(
                f"Corpus folder not found: {corpus_folder}"
            )

        self.corpus_folder = corpus_folder
        self._create_vectorizers_and_vocabs()

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
