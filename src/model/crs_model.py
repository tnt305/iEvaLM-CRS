import sys
from typing import Any, Dict

sys.path.append("..")

from src.model.BARCOR import BARCOR
from src.model.CHATGPT import CHATGPT
from src.model.KBRD import KBRD
from src.model.UNICRS import UNICRS

name2class = {
    "kbrd": KBRD,
    "barcor": BARCOR,
    "unicrs": UNICRS,
    "chatgpt": CHATGPT,
}


class CRSModel:
    def __init__(self, crs_model, *args, **kwargs) -> None:
        model_class = name2class[crs_model]
        self.crs_model = model_class(*args, **kwargs)

    def get_rec(self, conv_dict):
        return self.crs_model.get_rec(conv_dict)

    def get_conv(self, conv_dict):
        return self.crs_model.get_conv(conv_dict)

    def get_response(
        self, conv_dict: Dict[str, Any], id2entity: Dict[int, str], **kwargs
    ) -> str:
        """Generates a response given a conversation context.

        Args:
            conv_dict: Conversation context.
            id2entity: Mapping from entity id to entity name.

        Returns:
            Generated response.
        """
        return self.crs_model.get_response(conv_dict, id2entity, **kwargs)

    def get_choice(self, gen_inputs, option, state, conv_dict=None):
        return self.crs_model.get_choice(gen_inputs, option, state, conv_dict)
