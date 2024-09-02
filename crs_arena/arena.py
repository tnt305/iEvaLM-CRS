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

import json
import logging
import os
import time
from typing import Dict

import streamlit as st

from crs_arena.battle_manager import (
    CONVERSATION_COUNTS,
    get_crs_fighters,
    get_unique_user_id,
)
from crs_arena.crs_fighter import CRSFighter
from crs_arena.utils import execute_sql_query

# A message is a dictionary with two keys: role and message.
Message = Dict[str, str]

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CONVERSATION_LOG_DIR = "data/arena/conversation_logs/"
os.makedirs(CONVERSATION_LOG_DIR, exist_ok=True)


# Database setup
# Create the votes table if it doesn't exist
execute_sql_query(
    "CREATE TABLE IF NOT EXISTS votes ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
    "user_id TEXT, "
    "crs1 TEXT, "
    "crs2 TEXT, "
    "vote TEXT, "
    "feedback TEXT"
    ");",
    {},
)


# Callbacks
def record_vote(vote: str) -> None:
    """Record the user's vote in the database.

    Args:
        vote: Voted CRS model name.
    """
    user_id = st.session_state["user_id"]
    crs1_model: CRSFighter = st.session_state["crs1"]
    crs2_model: CRSFighter = st.session_state["crs2"]
    r = execute_sql_query(
        "INSERT INTO votes (user_id, crs1, crs2, vote) VALUES "
        "(:user_id, :crs1, :crs2, :vote) RETURNING id;",
        {
            "user_id": user_id,
            "crs1": crs1_model.name,
            "crs2": crs2_model.name,
            "vote": vote,
        },
    )
    last_row_id = r[0][0]
    feedback_dialog(row_id=last_row_id)


def record_feedback(feedback: str, row_id: int) -> None:
    """Record the user's feedback in the database and restart the app.

    Args:
        feedback: User's feedback.
        vote: Voted CRS model name.
        crs_models: Tuple of CRS model names.
        user_id: Unique user ID.
    """
    execute_sql_query(
        "UPDATE votes SET feedback = :feedback WHERE id = :row_id;",
        {"feedback": feedback, "row_id": row_id},
    )


def end_conversation(crs: CRSFighter) -> None:
    """Ends the conversation with given CRS model.

    Records the conversation in the logs and moves either to the next CRS or
    to the voting section.

    Args:
        crs: CRS model.
    """
    messages = st.session_state[f"messages_{crs.fighter_id}"]
    user_id = st.session_state["user_id"]
    logger.info(f"User {user_id} ended conversation with {crs.name}.")
    with open(
        os.path.join(CONVERSATION_LOG_DIR, f"{user_id}_{crs.name}.json"), "a"
    ) as f:
        json.dump(messages, f)

    # Update the conversation count
    CONVERSATION_COUNTS[crs.name] += 1

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
    response = crs.reply(
        input_message=message,
        history=st.session_state[f"messages_{crs.fighter_id}"],
    )
    # response = "CRS response for testing purposes."
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

st.header("Rules")
st.write(
    "* Chat with each CRS (one after the other) to get movie recommendation. "
    "Once you finish chatting with the CRS, click on the 'Done with CRS 1/2' "
    "button.\n"
    "* Vote on which CRS you prefer.\n"
    "* (Optional) Provide more detailed feedback after voting.\n"
)

# Side-by-Side Battle
st.header("Side-by-Side Battle")
st.write("Let's start the battle!\n")

# Initialize the chat messages
if "messages_1" not in st.session_state:
    st.session_state["messages_1"] = []
if "messages_2" not in st.session_state:
    st.session_state["messages_2"] = []

if "crs1_enabled" not in st.session_state:
    st.session_state["crs1_enabled"] = True
if "crs2_enabled" not in st.session_state:
    st.session_state["crs2_enabled"] = False
if "vote_enabled" not in st.session_state:
    st.session_state["vote_enabled"] = False

col_crs1, col_crs2 = st.columns(2)

# CRS 1
with col_crs1:
    with st.container(border=True):
        st.write("CRS 1")
        # Display the chat history
        messages_crs1 = st.container(height=350, border=False)
        for message in st.session_state["messages_1"]:
            messages_crs1.chat_message(message["role"]).write(
                message["message"]
            )

        if prompt1 := st.chat_input(
            "Send a message to CRS 1",
            key="prompt_crs1",
            disabled=not st.session_state["crs1_enabled"],
        ):
            # Display the user's message
            messages_crs1.chat_message("user").write(prompt1)

            # Get the CRS response
            response_crs1 = messages_crs1.chat_message(
                "assistant"
            ).write_stream(get_crs_response(st.session_state["crs1"], prompt1))

            # Add turn to chat history
            st.session_state["messages_1"].append(
                {"role": "user", "message": prompt1}
            )
            st.session_state["messages_1"].append(
                {"role": "assistant", "message": response_crs1}
            )
        st.button(
            "Done with CRS 1",
            use_container_width=True,
            key="end_crs1",
            on_click=end_conversation,
            kwargs={"crs": st.session_state["crs1"]},
            disabled=not st.session_state["crs1_enabled"],
        )

# CRS 2
with col_crs2:
    with st.container(border=True):
        st.write("CRS 2")
        # Display the chat history
        messages_crs2 = st.container(height=350, border=False)
        for message in st.session_state["messages_2"]:
            messages_crs2.chat_message(message["role"]).write(
                message["message"]
            )

        if prompt2 := st.chat_input(
            "Send a message to CRS 2",
            key="prompt_crs2",
            disabled=not st.session_state["crs2_enabled"],
        ):
            # Display the user's message
            messages_crs2.chat_message("user").write(prompt2)

            # Get the CRS response
            response_crs2 = messages_crs2.chat_message(
                "assistant"
            ).write_stream(get_crs_response(st.session_state["crs2"], prompt2))

            # Add turn to chat history
            st.session_state["messages_2"].append(
                {"role": "user", "message": prompt2}
            )
            st.session_state["messages_2"].append(
                {"role": "assistant", "message": response_crs2}
            )

        st.button(
            "Done with CRS 2",
            use_container_width=True,
            key="end_crs2",
            on_click=end_conversation,
            kwargs={"crs": st.session_state["crs2"]},
            disabled=not st.session_state["crs2_enabled"],
        )

# Feedback section
container = st.container()
container.subheader("Declare the winner!", anchor="vote")
container_col1, container_col2 = container.columns(2)
container_col1.button(
    "CRS 1",
    use_container_width=True,
    key="crs1_wins",
    on_click=record_vote,
    kwargs={"vote": st.session_state["crs1"].name},
    disabled=not st.session_state["vote_enabled"],
)
container_col2.button(
    "CRS 2",
    use_container_width=True,
    key="crs2_wins",
    on_click=record_vote,
    kwargs={"vote": st.session_state["crs2"].name},
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
