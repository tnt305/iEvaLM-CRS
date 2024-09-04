"""Utility functions for data preprocessing."""

import json
import re
from typing import Any, Dict, List

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("stopwords")

DEFAULT_ITEM_PLACEHOLDER = "ITEM_ID"


def remove_stopwords(utterance: str) -> str:
    """Removes stopwords from an utterance.

    Args:
        utterance: Input utterance.

    Returns:
        Utterance without stopwords.
    """
    tokens = word_tokenize(utterance)
    filtered_tokens = [
        token for token in tokens if token not in stopwords.words()
    ]
    return " ".join(filtered_tokens)


def expand_contractions(utterance: str) -> str:
    """Expands contractions in an utterance.

    Args:
        utterance: Input utterance.

    Returns:
        Utterance with expanded contractions.
    """
    contractions = json.load(open("data/crb_crs/contractions.json", "r"))
    for word in utterance.split():
        if word.lower() in contractions:
            utterance = utterance.replace(word, contractions[word.lower()])
    return utterance


def redial_replace_movie_ids(
    utterance: str, movie_placeholder: str = DEFAULT_ITEM_PLACEHOLDER
) -> str:
    """Replaces movie ids with a placeholder in utterance from ReDial dataset.

    Args:
        utterance: Input utterance.
        movie_placeholder: Placeholder for movie id.

    Returns:
        Utterance with movie ids replaced by placeholder.
    """
    if "@" in utterance:
        movie_ids = re.findall(r"@\S+", utterance)
        if movie_ids:
            for movie_id in movie_ids:
                utterance = utterance.replace(movie_id, movie_placeholder)
    return utterance


def opendialkg_replace_items(
    text: str,
    items: List[str],
    item_placeholder: str = DEFAULT_ITEM_PLACEHOLDER,
):
    """Replaces items with a placeholder in utterance from OpenDialKG dataset.

    Args:
        text: Input utterance.
        items: List of items in the utterance (taken from dataset).
        item_placeholder: Placeholder for item.

    Returns:
        Utterance with items replaced by placeholder.
    """
    for item in items:
        text = text.replace(item, item_placeholder)
    return text


def preprocess_utterance(
    utterance: Dict[str, Any],
    dataset: str,
    item_placeholder: str = DEFAULT_ITEM_PLACEHOLDER,
    no_stopwords: bool = True,
) -> str:
    """Preprocesses an utterance.

    Preprocessing includes lowercasing, stripping, replacing item id with a
    palceholder, converting contractions to full form, and removing stopwords.

    Args:
        utterance: Input utterance.
        dataset: Name of the origin dataset.
        item_placeholder: Placeholder for item id.
        stopwords: Whether to remove stopwords.

    Raises:
        ValueError: If dataset is not supported.

    Returns:
        Preprocessed utterance.
    """
    processed_utterance = utterance.get("text").lower().strip()

    if dataset == "redial":
        processed_utterance = redial_replace_movie_ids(
            processed_utterance, item_placeholder
        )
    elif dataset == "opendialkg":
        processed_utterance = opendialkg_replace_items(
            processed_utterance, utterance.get("items"), item_placeholder
        )
    else:
        raise ValueError(f"Dataset {dataset} not supported.")

    processed_utterance = expand_contractions(processed_utterance)
    if no_stopwords:
        processed_utterance = remove_stopwords(processed_utterance)

    if processed_utterance == "":
        processed_utterance = "**"

    return processed_utterance


def get_preference_keywords(domain: str) -> List[str]:
    """Returns a list of preference keywords.

    Args:
        domain: Domain name.

    Raises:
        ValueError: If the domain is not supported.
    """
    movies_preference_keywords = [
        "scary",
        "horror",
        "pixar",
        "graphic",
        "classic",
        "comedy",
        "kids",
        "funny",
        "disney",
        "comedies",
        "action",
        "family",
        "adventure",
        "crime",
        "fantasy",
        "thriller",
        "scifi",
        "documentary",
        "science fiction",
        "drama",
        "romance",
        "romances",
        "romantic",
        "mystery",
        "mysteries",
        "history",
        "no preference",
        "suspense",
    ]
    if domain == "movies":
        return movies_preference_keywords
    elif domain == "movies_books":
        return (
            movies_preference_keywords + []
        )  # TOOD: Add more keywords related to books
    raise ValueError(f"Domain not supported: {domain}")
