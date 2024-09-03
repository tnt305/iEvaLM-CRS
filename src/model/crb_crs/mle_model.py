"""N-gram Maximum Likelihood Probabilistic Language Model.

Supports n-gram up to 5-gram.
"""

from __future__ import annotations

import math as calc
import os
import pickle
from collections import Counter, defaultdict
from typing import List

from nltk.util import ngrams


class NGramMLE:
    def __init__(self, n: int = 1, corpus_file: str = None) -> None:
        """Initializes the model.

        Args:
            n: n-gram order. Defaults to 1.
            corpus_file: File containing the corpus words. Defaults to None.

        Raises:
            ValueError: If n is not in [1, 2, 3, 4, 5].
            FileNotFoundError: If corpus_file is defined but not found.
        """
        if n not in [1, 2, 3, 4, 5]:
            raise ValueError("n must be in [1, 2, 3, 4, 5]")
        if corpus_file and not os.path.exists(corpus_file):
            raise FileNotFoundError(f"Corpus file not found: {corpus_file}")

        self.n = n
        self.corpus_file = corpus_file

        self.ngrams = defaultdict(Counter)

    def _read_corpus(self) -> List[str]:
        """Reads the corpus from the file.

        Returns:
            List of words in the corpus.
        """
        with open(self.corpus_file, "r") as f:
            return f.read().split("\n")

    def create_ngrams(self):
        """Creates n-grams from the corpus."""
        corpus_words = self._read_corpus()
        self.total_words = len(corpus_words)

        self.ngrams[1] = Counter(corpus_words)
        for i in range(self.n - 1):
            self.ngrams[i + 2] = Counter(ngrams(corpus_words, i + 2))

    def probability(
        self, ngram: str, higher_order_ngram: str = "", n: int = 1
    ) -> float:
        """Computes maximum likelihood probability.

        Args:
            ngram: n-gram.
            higher_order_ngram: Higher order n-gram. Defaults to "".
            n: n-gram order. Defaults to 1.

        Returns:
            Maximum likelihood probability.
        """
        if n == 1:
            return calc.log(
                (self.ngrams[1][ngram] + 1) / (self.total_words + len(self.ngrams[1]))
            )

        assert n <= self.n, f"n must be less than or equal to {self.n}"
        return calc.log(
            (self.ngrams[n][higher_order_ngram] + 1)
            / (self.ngrams[n - 1][ngram] + len(self.ngrams[1]))
        )

    def sentence_probability(self, sentence: str, n: int = 1) -> float:
        """Computes cumulative n-gram ML probability of a sentence.

        Args:
            sentence: Sentence.
            n: n-gram order. Defaults to 1.

        Returns:
            Cumulative n-gram maximum likelihood probability.
        """
        words = sentence.lower().split()
        cumulative_prob = 0

        if n == 1:
            for word in words:
                cumulative_prob += self.probability(word)
            return cumulative_prob

        for i, word in enumerate(words):
            if i >= len(words) - n - 1:
                break
            cumulative_prob += self.probability(
                " ".join(words[i : i + n - 2]),
                " ".join(words[i : i + n - 1]),
                n,
            )

        return cumulative_prob

    @classmethod
    def load(cls, model_file: str) -> NGramMLE:
        """Loads the model from a file.

        Args:
            model_file: File containing the model.

        Returns:
            Loaded model.
        """
        return pickle.load(open(model_file, "rb"))

    def save(self, model_file: str) -> None:
        """Saves the model to a file.

        Args:
            model_file: File to save the model.
        """
        pickle.dump(self, open(model_file, "wb"))
