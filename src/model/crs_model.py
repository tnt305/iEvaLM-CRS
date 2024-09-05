import sys
from typing import Any, Dict, List, Tuple

sys.path.append("..")

from src.model.BARCOR import BARCOR
from src.model.CHATGPT import CHATGPT
from src.model.CRB_CRS import CRBCRSModel
from src.model.KBRD import KBRD
from src.model.UNICRS import UNICRS

name2class = {
    "kbrd": KBRD,
    "barcor": BARCOR,
    "unicrs": UNICRS,
    "chatgpt": CHATGPT,
    "crbcrs": CRBCRSModel,
}


class CRSModel:
    def __init__(self, crs_model, *args, **kwargs) -> None:
        model_class = name2class[crs_model]
        self.crs_model = model_class(*args, **kwargs)

    def get_rec(self, conv_dict: Dict[str, Any]):
        """Generates recommendations given a conversation context."""
        return self.crs_model.get_rec(conv_dict)

    def get_conv(self, conv_dict: Dict[str, Any]):
        """Generates utterance given a conversation context."""
        return self.crs_model.get_conv(conv_dict)

    def get_response(
        self,
        conv_dict: Dict[str, Any],
        id2entity: Dict[int, str],
        options: Tuple[str, Dict[str, str]],
        state: List[float],
        **kwargs
    ) -> Tuple[str, List[float]]:
        """Generates a response given a conversation context.

        The method is based on the logic of the ask mode (i.e., see
        `scripts/ask.py`). It consists of two steps: (1) choose to either
        recommend items or generate a response, and (2) execute the chosen
        step.

        Args:
            conv_dict: Conversation context.
            id2entity: Mapping from entity id to entity name.
            options: Prompt with options and dictionary of options.
            state: State of the option choices.

        Returns:
            Generated response and updated state.
        """
        return self.crs_model.get_response(
            conv_dict, id2entity, options, state, **kwargs
        )

    def get_choice(self, gen_inputs, option, state, conv_dict=None):
        """Generates a choice between options given a conversation context."""
        return self.crs_model.get_choice(gen_inputs, option, state, conv_dict)
