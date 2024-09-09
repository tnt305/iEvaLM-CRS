"""Streamlit app for side-by-side battle of two CRSs.

The goal of this application is for a user to have a conversation with two
conversational recommender systems (CRSs) and vote on which one they prefer.
All the conversations are recorded and saved for future analysis. When the user
arrives on the app, they are assigned a unique ID. Then, two CRSs are chosen
for the battle depending on the number of conversations already recorded (i.e.,
the CRSs with the least number of conversations are chosen). The user is given
a few constraints on the items they are looking for, after they interact with
the CRSs one after the other. The user can then vote on which CRS they prefer,
upon voting a pop-up will appear giving the user the option to provide a more
detailed feedback. Once the vote is submitted, all data is logged and the app
restarts for a new battle.

The app is composed of four sections:
1. Title/Introduction
2. Rules
3. Side-by-Side Battle
4. Feedback
"""

import asyncio
import json
import logging
import os
import time
from copy import deepcopy
from datetime import datetime
from typing import Dict, List

import streamlit as st
from battle_manager import (
    CONVERSATION_COUNTS,
    cache_fighters,
    get_crs_fighters,
    get_unique_user_id,
)
from crs_fighter import CRSFighter
from utils import (
    download_and_extract_item_embeddings,
    download_and_extract_models,
    upload_conversation_logs_to_hf,
    upload_feedback_to_gsheet,
)

from src.model.crb_crs.recommender import *

# A message is a dictionary with two keys: role and message.
Message = Dict[str, str]

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Download models and data externally stored if not already downloaded
if not os.path.exists("data/models"):
    logger.info("Downloading models...")
    download_and_extract_models()
if not os.path.exists("data/embed_items"):
    logger.info("Downloading item embeddings...")
    download_and_extract_item_embeddings()

# Create the conversation logs directory
CONVERSATION_LOG_DIR = "data/arena/conversation_logs/"
os.makedirs(CONVERSATION_LOG_DIR, exist_ok=True)


# Callbacks
def record_vote(vote: str) -> None:
    """Record the user's vote in the database.

    Args:
        vote: Voted CRS model name.
    """
    user_id = st.session_state["user_id"]
    crs1_model: CRSFighter = st.session_state["crs1"]
    crs2_model: CRSFighter = st.session_state["crs2"]
    last_row_id = str(datetime.now())
    asyncio.run(
        upload_feedback_to_gsheet(
            {
                "id": last_row_id,
                "user_id": user_id,
                "crs1": crs1_model.name,
                "crs2": crs2_model.name,
                "vote": vote,
            },
            worksheet="votes",
        )
    )
    feedback_dialog(row_id=last_row_id)


def record_feedback(feedback: str, row_id: int) -> None:
    """Record the user's feedback in the database and restart the app.

    Args:
        feedback: User's feedback.
        vote: Voted CRS model name.
        crs_models: Tuple of CRS model names.
        user_id: Unique user ID.
    """
    asyncio.run(
        upload_feedback_to_gsheet(
            {"id": row_id, "feedback": feedback}, "feedback"
        )
    )


def end_conversation(crs: CRSFighter, sentiment: str) -> None:
    """Ends the conversation with given CRS model.

    Records the conversation in the logs and moves either to the next CRS or
    to the voting section.

    Args:
        crs: CRS model.
        sentiment: User's sentiment (frustrated or satisfied).
    """
    messages: List[Message] = deepcopy(
        st.session_state[f"messages_{crs.fighter_id}"]
    )
    messages.append({"role": "metadata", "sentiment": sentiment})
    user_id = st.session_state["user_id"]
    logger.info(f"User {user_id} ended conversation with {crs.name}.")
    log_file_path = os.path.join(
        CONVERSATION_LOG_DIR, f"{user_id}_{crs.name}.json"
    )
    with open(log_file_path, "a") as f:
        json.dump(messages, f)

    # Update the conversation count
    CONVERSATION_COUNTS[crs.name] += 1

    # Asynchronously save the conversation logs to Hugging Face Hub
    asyncio.run(
        upload_conversation_logs_to_hf(
            log_file_path, f"conversation_logs/{user_id}_{crs.name}.json"
        )
    )

    if crs.fighter_id == 1:
        # Disable the chat interface for the first CRS
        st.session_state["crs1_enabled"] = False
        # Enable the chat interface for the second CRS
        st.session_state["crs2_enabled"] = True
    elif crs.fighter_id == 2:
        # Disable the chat interface for the second CRS
        st.session_state["crs2_enabled"] = False
        # Enable the voting section
        st.session_state["vote_enabled"] = True
        # Scroll to the voting section


@st.fragment
def chat_col(crs_id: int, color: str):
    """Chat column for the CRS model.

    Args:
        crs_id: CRS model ID (either 1 or 2).
        color: Color of the CRS model (red or large_blue).
    """
    with st.container(border=True):
        st.write(f":{color}_circle: CRS {crs_id}")
        # Display the chat history
        messages_crs = st.container(height=350, border=False)
        for message in st.session_state[f"messages_{crs_id}"]:
            messages_crs.chat_message(message["role"]).write(
                message["message"]
            )

        if prompt := st.chat_input(
            f"Send a message to CRS {crs_id}",
            key=f"prompt_crs{crs_id}",
            disabled=not st.session_state[f"crs{crs_id}_enabled"],
        ):
            # Display the user's message
            messages_crs.chat_message("user").write(prompt)
            # Add user's message to chat history
            st.session_state[f"messages_{crs_id}"].append(
                {"role": "user", "message": prompt}
            )
            # Get the CRS response
            response_crs = messages_crs.chat_message("assistant").write_stream(
                get_crs_response(st.session_state[f"crs{crs_id}"], prompt)
            )

            # Add CRS response to chat history
            st.session_state[f"messages_{crs_id}"].append(
                {"role": "assistant", "message": response_crs}
            )

        frustrated_col, satisfied_col = st.columns(2)
        if frustrated_col.button(
            ":rage: Frustrated",
            use_container_width=True,
            key=f"end_frustated_crs{crs_id}",
            on_click=end_conversation,
            kwargs={
                "crs": st.session_state[f"crs{crs_id}"],
                "sentiment": "frustrated",
            },
            disabled=not st.session_state[f"crs{crs_id}_enabled"],
        ):
            st.rerun()
        if satisfied_col.button(
            ":heavy_check_mark: Satisfied",
            use_container_width=True,
            key=f"end_satisfied_crs{crs_id}",
            on_click=end_conversation,
            kwargs={
                "crs": st.session_state[f"crs{crs_id}"],
                "sentiment": "satisfied",
            },
            disabled=not st.session_state[f"crs{crs_id}_enabled"],
        ):
            st.rerun()


def get_crs_response(crs: CRSFighter, message: str):
    """Gets the CRS response for the given message.

    This method sends a POST request to the CRS model including the history of
    the conversation and the user's message.

    Args:
        crs: CRS model.
        message: User's message.

    Yields:
        Words from the CRS response.
    """
    history = deepcopy(st.session_state[f"messages_{crs.fighter_id}"])
    history = history[:-1]  # Remove the last message (user's message)

    response, state = crs.reply(
        input_message=message,
        history=st.session_state[f"messages_{crs.fighter_id}"],
        options_state=st.session_state.get(f"state_{crs.fighter_id}", []),
    )
    st.session_state[f"state_{crs.fighter_id}"] = state
    for word in response.split():
        yield f"{word} "
        time.sleep(0.05)


@st.dialog("Your vote has been submitted!")
def feedback_dialog(row_id: int) -> None:
    """Pop-up dialog to provide feedback after voting.

    Feedback is optional and can be used to provide more detailed information
    about the user's vote.

    Args:
        row_id: Unique row ID of the vote.
    """
    feedback_text = st.text_area(
        "(Optional) You can provide more detailed feedback below:"
    )
    if st.button("Finish", use_container_width=True):
        record_feedback(feedback_text, row_id)
        # Restart the app
        st.session_state.clear()
        st.rerun()


# Streamlit app
st.set_page_config(page_title="CRS Arena", layout="wide")

# Cache some CRS fighters at startup
cache_fighters(n=2)

# Battle setup
if "user_id" not in st.session_state:
    st.session_state["user_id"] = get_unique_user_id()
    st.session_state["crs1"], st.session_state["crs2"] = get_crs_fighters()


# Introduction
st.title(":gun: CRS Arena")
st.write(
    "Welcome to the CRS Arena! Here you can have a conversation with two "
    "conversational recommender systems (CRSs) and vote on which one you "
    "prefer."
)

st.header(":page_with_curl: Rules")
st.write(
    "* Chat with each CRS (one after the other) to get movie recommendations "
    "up until you feel statisfied or frustrated.\n"
    "* Try to send several messages to each CRS to get a better sense of their "
    "capabilities. Don't quit after the first message!\n"
    "* To finish chatting with a CRS, click on the button corresponding to "
    "your feeling: frustrated or satisfied.\n"
    "* Vote on which CRS you prefer or declare a tie.\n"
    "* (Optional) Provide more detailed feedback after voting.\n"
)

# Side-by-Side Battle
st.header(":point_down: Side-by-Side Battle")
st.write("Let's start the battle!")

# Initialize the chat messages
if "messages_1" not in st.session_state:
    st.session_state["messages_1"] = []
    st.session_state["state_1"] = None
if "messages_2" not in st.session_state:
    st.session_state["messages_2"] = []
    st.session_state["state_2"] = None

if "crs1_enabled" not in st.session_state:
    st.session_state["crs1_enabled"] = True
if "crs2_enabled" not in st.session_state:
    st.session_state["crs2_enabled"] = False
if "vote_enabled" not in st.session_state:
    st.session_state["vote_enabled"] = False

col_crs1, col_crs2 = st.columns(2)

# CRS 1
with col_crs1:
    chat_col(1, "red")


# CRS 2
with col_crs2:
    chat_col(2, "large_blue")


# Feedback section
container = st.container()
container.subheader(":trophy: Declare the winner!", anchor="vote")
container_col1, container_col2, container_col3 = container.columns(3)
container_col1.button(
    ":red_circle: CRS 1",
    use_container_width=True,
    key="crs1_wins",
    on_click=record_vote,
    kwargs={"vote": st.session_state["crs1"].name},
    disabled=not st.session_state["vote_enabled"],
)
container_col2.button(
    ":large_blue_circle: CRS 2",
    use_container_width=True,
    key="crs2_wins",
    on_click=record_vote,
    kwargs={"vote": st.session_state["crs2"].name},
    disabled=not st.session_state["vote_enabled"],
)
container_col3.button(
    "Tie",
    use_container_width=True,
    key="tie",
    on_click=record_vote,
    kwargs={"vote": "tie"},
    disabled=not st.session_state["vote_enabled"],
)

# Terms of service
st.header("Terms of Service")
st.write(
    "By using this application, you agree to the following terms of service:\n"
    "The service is a research platform. Please do not upload any private "
    "information in the chat. The service collects the chat data and the user's"
    " vote, which may be released under a Creative Commons Attribution (CC-BY) "
    "or a similar license."
)
