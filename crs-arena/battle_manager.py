"""Battle Manager module.

Contains helper functions to select fighters for a battle and generate
unique user ids.
"""

import uuid
from collections import defaultdict
from typing import Tuple

CRS_MODELS = {
    "kbrd": "localhost:5000",
    "unicrs": "localhost:5001",
    "barcor": "localhost:5002",
    "chatgpt": "localhost:5003",
    "rb-crs": "localhost:5004",
}

CONVERSATION_COUNTS = defaultdict(int).fromkeys(CRS_MODELS.keys(), 0)


def get_crs_fighters() -> Tuple[str, str]:
    """Selects two CRS models for a battle.

    The selection is based on the number of conversations collected per model.
    The ones with the least conversations are selected.

    Returns:
        CRS models to battle.
    """
    pair = sorted(CONVERSATION_COUNTS.items(), key=lambda x: x[1])[:2]
    return pair[0][0], pair[1][0]


def get_unique_user_id() -> str:
    """Generates a unique user id.

    Returns:
        Unique user id.
    """
    return str(uuid.uuid4())
