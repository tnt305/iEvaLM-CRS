"""Script to prepare the ReDial data for CRB-CRS model."""

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from src.model.crb_crs.retriever.retriever import (
    CONV_PREFIX,
    CRS_PREFIX,
    USER_PREFIX,
)
from src.model.crb_crs.utils_preprocessing import preprocess_utterance

ParsedDialogue = List[str]


def read_jsonl_data(path: str) -> List[Dict[str, Any]]:
    """Reads data from a jsonl file.

    Args:
        path: Path to the jsonl file.

    Returns:
        List of dictionaries.
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data


def parse_dialogue(
    dialogue: Dict[str, Any], idx: int
) -> Tuple[ParsedDialogue, ParsedDialogue, ParsedDialogue]:
    """Parses a dialogue.

    Produce three types of parsed dialogues:
    1. Parsed dialogue with original utterances.
    2. Parsed dialogue with preprocessed utterances but incluiding stopwords.
    3. Parsed dialogue with preprocessed utterances without stopwords.

    Args:
        dialogue: Dialogue.
        idx: Index of the dialogue.

    Returns:
        List of dialogue utterances with participant prefix.
    """
    parsed_dialogue_original = [f"{CONV_PREFIX} {idx}"]
    parsed_dialogue_preprocessed = [f"{CONV_PREFIX} {idx}"]
    parsed_dialogue_preprocessed_no_stopwords = [f"{CONV_PREFIX} {idx}"]

    user_id = dialogue.get("initiatorWorkerId")
    system_id = dialogue.get("respondentWorkerId")

    for message in dialogue.get("messages", []):
        sender_id = message.get("senderWorkerId")
        utterance = message.get("text")
        preprocessed_utterance = preprocess_utterance(
            {"text": utterance}, "redial", no_stopwords=False
        )
        preprocessed_utterance_no_stopwords = preprocess_utterance(
            {"text": utterance}, "redial", no_stopwords=True
        )

        if sender_id == user_id:
            parsed_dialogue_original.append(f"{USER_PREFIX} {utterance}")
        elif sender_id == system_id:
            parsed_dialogue_original.append(f"{CRS_PREFIX} {utterance}")

        parsed_dialogue_preprocessed.append(preprocessed_utterance)
        parsed_dialogue_preprocessed_no_stopwords.append(
            preprocessed_utterance_no_stopwords
        )
    return (
        parsed_dialogue_original,
        parsed_dialogue_preprocessed,
        parsed_dialogue_preprocessed_no_stopwords,
    )


def parse_dialogues(
    dialogues: List[Dict[str, Any]]
) -> Tuple[ParsedDialogue, ParsedDialogue, ParsedDialogue]:
    """Parses dialogues.

    Args:
        dialogues: List of dialogues.

    Returns:
        List of parsed dialogues.
    """
    parsed_dialogues_original = []
    parsed_dialogues_preprocessed = []
    parsed_dialogues_preprocessed_no_stopwords = []

    for i, dialogue in enumerate(tqdm(dialogues)):
        (
            parsed_dialogue_original,
            parsed_dialogue_preprocessed,
            parsed_dialogue_preprocessed_no_stopwords,
        ) = parse_dialogue(dialogue, i)

        parsed_dialogues_original.extend(parsed_dialogue_original)
        parsed_dialogues_preprocessed.extend(parsed_dialogue_preprocessed)
        parsed_dialogues_preprocessed_no_stopwords.extend(
            parsed_dialogue_preprocessed_no_stopwords
        )

    return (
        parsed_dialogues_original,
        parsed_dialogues_preprocessed,
        parsed_dialogues_preprocessed_no_stopwords,
    )


def save_parsed_dialogues(parsed_dialogues: List[str], path: str) -> None:
    """Saves parsed dialogues to a file.

    Args:
        parsed_dialogues: List of parsed dialogues.
        path: Path to the output file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for utterance in parsed_dialogues:
            f.write(f"{utterance}\n")


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare ReDial data for CRB-CRS model."
    )
    parser.add_argument(
        "--redial_folder",
        type=str,
        default="data/redial",
        help="Path to folder with ReDial dialogues.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data/crb_crs/",
        help="Path to output folder.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    raw_dialogues = []
    for file in ["train_data.jsonl", "valid_data.jsonl", "test_data.jsonl"]:
        if os.path.exists(os.path.join(args.redial_folder, file)):
            raw_dialogues.extend(
                read_jsonl_data(os.path.join(args.redial_folder, file))
            )

    logging.info(f"Loaded {len(raw_dialogues)} dialogues.")

    (
        parsed_dialogue_original,
        parsed_dialogue_preprocessed,
        parsed_dialogue_preprocessed_no_stopwords,
    ) = parse_dialogues(raw_dialogues)

    logging.info("Finished parsing dialogues.")

    save_parsed_dialogues(
        parsed_dialogue_original,
        os.path.join(
            args.output_folder, "parsed_redial_dialogues_original.txt"
        ),
    )
    save_parsed_dialogues(
        parsed_dialogue_preprocessed,
        os.path.join(
            args.output_folder, "parsed_redial_dialogues_preprocessed.txt"
        ),
    )
    save_parsed_dialogues(
        parsed_dialogue_preprocessed_no_stopwords,
        os.path.join(
            args.output_folder,
            "parsed_redial_dialogues_preprocessed_no_stopwords.txt",
        ),
    )

    logging.info(f"Saved parsed dialogues to {args.output_folder}.")
