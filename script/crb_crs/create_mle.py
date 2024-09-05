"""Script to create MLE model for retriever component of CRB-CRS.

For ReDial, use the following command:
python -m script.crb_crs.create_mle \
    --corpus_file data/redial/GT_corpus_tokens.txt \
    --output_file data/models/crb_crs/mle_model.pkl
"""

import argparse
import logging
import os

from src.model.crb_crs.retriever.mle_model import NGramMLE


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create MLE model for retriever component of CRB-CRS."
    )
    parser.add_argument(
        "--corpus_file",
        type=str,
        required=True,
        help="Path to the corpus file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the created MLE model.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2,
        help="Maximum n-gram order. Defaults to 2.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Creates MLE model for retriever component of CRB-CRS.

    Args:
        args: Command line arguments.
    """
    model = NGramMLE(args.n, args.corpus_file)

    model.create_ngrams()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    model.save(args.output_file)
    logging.info(f"MLE model saved at {args.output_file}.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(parse_args())
