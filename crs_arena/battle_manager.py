"""Battle Manager module.

Contains helper functions to select fighters for a battle and generate
unique user ids.
"""

import itertools
import logging
import random
import uuid
from collections import defaultdict
from typing import Optional, Tuple

from crs_fighter import CRSFighter
from utils import get_crs_model

# CRS models with their configuration files.
CRS_MODELS = {
    "crbcrs_redial": "data/arena/crs_config/CRB_CRS/crb_crs_redial.yaml",
    "kbrd_redial": "data/arena/crs_config/KBRD/kbrd_redial.yaml",
    "unicrs_redial": "data/arena/crs_config/UniCRS/unicrs_redial.yaml",
    "kbrd_opendialkg": "data/arena/crs_config/KBRD/kbrd_opendialkg.yaml",
    "chatgpt_redial": "data/arena/crs_config/ChatGPT/chatgpt_redial.yaml",
    "barcor_opendialkg": "data/arena/crs_config/BARCOR/barcor_opendialkg.yaml",
    "unicrs_opendialkg": "data/arena/crs_config/UniCRS/unicrs_opendialkg.yaml",
    "barcor_redial": "data/arena/crs_config/BARCOR/barcor_redial.yaml",
    "chatgpt_opendialkg": (
        "data/arena/crs_config/ChatGPT/chatgpt_opendialkg.yaml"
    ),
}

CONVERSATION_COUNTS = defaultdict(int).fromkeys(CRS_MODELS.keys(), 0)


def get_crs_fighters() -> Tuple[CRSFighter, CRSFighter]:
    """Selects two CRS models for a battle.

    The selection is based on the number of conversations collected per model.
    The ones with the least conversations are selected.

    Raises:
        Exception: If there is an error selecting the fighters.

    Returns:
        CRS models to battle.
    """
    sorted_count = sorted(CONVERSATION_COUNTS.items(), key=lambda x: x[1])
    # Group models by conversation count.
    groups = [
        list(group)
        for _, group in itertools.groupby(sorted_count, key=lambda x: x[1])
    ]

    model_1, model_2 = None, None

    try:
        if len(groups[0]) >= 2:
            model_1 = groups[0].pop(random.randint(0, len(groups[0]) - 1))[0]
            model_2 = groups[0].pop(random.randint(0, len(groups[0]) - 1))[0]
        else:
            model_1 = groups[0].pop(random.randint(0, len(groups[0]) - 1))[0]
            model_2 = groups[1].pop(random.randint(0, len(groups[1]) - 1))[0]
    except Exception as e:
        logging.error(f"Error selecting CRS fighters: {e}")
        if model_1 is None:
            model_1 = sorted_count[0][0]
        if model_2 is None:
            model_2 = sorted_count[1][0]

    fighter1 = CRSFighter(1, model_1, CRS_MODELS[model_1])
    fighter2 = CRSFighter(2, model_2, CRS_MODELS[model_2])
    return fighter1, fighter2


def get_unique_user_id() -> str:
    """Generates a unique user id.

    Returns:
        Unique user id.
    """
    return str(uuid.uuid4())


def cache_fighters(n: Optional[int] = None) -> None:
    """Caches n CRS fighters.

    Args:
        n: Number of fighters to cache. If None, all fighters are cached.
    """
    logging.info(f"Caching {n} CRS fighters.")
    for i, (model_name, config_path) in enumerate(CRS_MODELS.items()):
        get_crs_model(model_name, config_path)
        if n is not None and i == n:
            break
