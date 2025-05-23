import json
import random
from typing import Dict, List, Optional, Tuple, Union

import torch
from rapidfuzz import fuzz, process
from torch import nn
from torch.nn import functional as F

special_tokens_dict = {"pad_token": "<|pad|>"}


def load_jsonl_data(file):
    data_list = []
    with open(file, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            data_list.append(data)
    return data_list


def simple_collate(batch):
    return batch


def sample_data(data_list, shot=1, debug=False, number_for_debug=320):
    if debug:
        data_list = data_list[:number_for_debug]

    if shot < 1:
        data_idx = random.sample(
            range(len(data_list)), int(len(data_list) * shot)
        )
        data_list = [data_list[idx] for idx in data_idx]
    elif shot > 1:
        data_idx = range(int(shot))
        data_list = [data_list[idx] for idx in data_idx]

    return data_list


def padded_tensor(
    items: List[Union[List[int], torch.LongTensor]],
    pad_id: int = 0,
    pad_tail: bool = True,
    device: torch.device = torch.device("cpu"),
    debug: bool = False,
    max_length: Optional[int] = None,
) -> torch.Tensor:
    # number of items
    n = len(items)
    # length of each item
    lens: List[int] = [len(item) for item in items]
    # max in time dimension
    t = max(max(lens), 1)
    if debug and max_length is not None:
        t = max(t, max_length)

    output = torch.full(
        (n, t), fill_value=pad_id, dtype=torch.long, device=device
    )

    for i, (item, length) in enumerate(zip(items, lens)):
        if length == 0:
            continue
        if not isinstance(item, torch.Tensor):
            item = torch.as_tensor(item, dtype=torch.long, device=device)
        if pad_tail:
            output[i, :length] = item
        else:
            output[i, t - length :] = item

    return output


class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x, mask=None):
        """

        Args:
            x (bs, seq_len, hs)
            mask (bs, seq_len): False for masked token.

        Returns:
            (bs, hs)
        """
        attn = self.attn(x)  # (bs, seq_len, 1)
        if mask is not None:
            attn += (~mask).unsqueeze(-1) * -1e4
        attn = F.softmax(attn, dim=-1)
        x = attn.transpose(1, 2) @ x  # (bs, 1, hs)
        x = x.squeeze(1)
        return x


def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].detach().clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


# dbpedia get entity
# def get_entity(text, SPOTLIGHT_CONFIDENCE):
#     DBPEDIA_SPOTLIGHT_ADDR = " http://0.0.0.0:2222/rest/annotate"
#     headers = {"accept": "application/json"}
#     params = {"text": text, "confidence": SPOTLIGHT_CONFIDENCE}

#     response = requests.get(DBPEDIA_SPOTLIGHT_ADDR, headers=headers, params=params)
#     response = response.json()
#     return (
#         [f"<{x['@URI']}>" for x in response["Resources"]]
#         if "Resources" in response
#         else []
#     )


# rapidfuzz get entity
def get_entity(text, entity_list):
    extractions = process.extract(
        text, entity_list, scorer=fuzz.WRatio, limit=20
    )
    extractions = [
        extraction[0] for extraction in extractions if extraction[1] >= 90
    ]
    return extractions


def get_options(dataset: str) -> Tuple[str, Dict[str, str]]:
    """Returns the possible options for a given dataset.

    Args:
        dataset: The dataset to get options for.

    Raises:
        ValueError: If the dataset is not supported.

    Returns:
        A tuple containing the prompt and a dictionary of options.
    """
    if "redial" in dataset:
        instructions = (
            "To recommend me items that I will accept, you can choose one of "
            "the following options.\nA: ask my preference for genre\nB: ask my "
            "preference for actor\nC: ask my preference for director\nD: I can "
            "directly give recommendations\nPlease enter the option character. "
            "Please only response a character."
        )
        options = {
            "A": {"attribute": "genre", "template": "What genre do you like?"},
            "B": {"attribute": "actor", "template": "Which star do you like?"},
            "C": {
                "attribute": "director",
                "template": "Which director do you like?",
            },
            "D": {"attribute": "recommend", "template": ""},
        }
        return instructions, options
    elif "opendialkg" in dataset:
        instructions = (
            "To recommend me items that I will accept, you can choose one of "
            "the following options.\nA: ask my preference for genre\nB: ask my "
            "preference for actor\nC: ask my preference for director\nD: ask "
            "my preference for writer\nE: I can directly give recommendations"
            "\nPlease enter the option character. Please only response a "
            "character."
        )
        options = {
            "A": {"attribute": "genre", "template": "What genre do you like?"},
            "B": {"attribute": "actor", "template": "Which star do you like?"},
            "C": {
                "attribute": "director",
                "template": "Which director do you like?",
            },
            "D": {
                "attribute": "writer",
                "template": "Which writer do you like?",
            },
            "E": {"attribute": "recommend", "template": ""},
        }
        return instructions, options

    raise ValueError(f"Dataset {dataset} is not supported.")
