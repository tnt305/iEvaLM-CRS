"""Utility functions for CRS Arena."""

import asyncio
import logging
import os
import sqlite3
import sys
import tarfile
from datetime import timedelta
from typing import Any, Dict, List

import openai
import pandas as pd
import streamlit as st
import wget
import yaml
from huggingface_hub import HfApi
from streamlit_gsheets import GSheetsConnection

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model.crs_model import CRSModel

# Initialize Hugging Face API
HF_API = HfApi(token=st.secrets.hf.hf_token)


@st.cache_resource(show_spinner="Loading CRS...", max_entries=5)
def get_crs_model(model_name: str, model_config_file: str) -> CRSModel:
    """Returns a CRS model.

    Args:
        model_name: Model name.
        model_config_file: Model configuration file.

    Raises:
        FileNotFoundError: If model configuration file is not found.

    Returns:
        CRS model.
    """
    logging.debug(f"Loading CRS model {model_name}.")
    if not os.path.exists(model_config_file):
        raise FileNotFoundError(
            f"Model configuration file {model_config_file} not found."
        )

    model_args = yaml.safe_load(open(model_config_file, "r"))

    if "chatgpt" in model_name:
        openai.api_key = st.secrets.openai.api_key

    # Extract crs model from name
    name = model_name.split("_")[0]

    return CRSModel(name, **model_args)


def execute_sql_query(query: str, params: Dict[str, str]) -> List[Any]:
    """Executes a SQL query with parameters.

    Args:
        query: SQL query.
        params: Dictionary of parameters.

    Returns:
        Output of the query.
    """
    connection = sqlite3.connect(st.secrets.db.vote_db)
    cursor = connection.cursor().execute(query, params)
    output = cursor.fetchall()
    connection.commit()
    return output


def download_and_extract_models() -> None:
    """Downloads the models folder from the server and extracts it."""
    logging.debug("Downloading models folder.")
    models_url = st.secrets.files.models_folder_url
    models_targz = "models.tar.gz"
    models_folder = "data/models/"
    try:
        wget.download(models_url, models_targz)

        logging.debug("Extracting models folder.")
        with tarfile.open(models_targz, "r:gz") as tar:
            tar.extractall(models_folder)

        os.remove(models_targz)
        logging.debug("Models folder downloaded and extracted.")
    except Exception as e:
        logging.error(f"Error downloading models folder: {e}")


def download_and_extract_item_embeddings() -> None:
    """Downloads the item embeddings folder from the server and extracts it."""
    logging.debug("Downloading item embeddings folder.")
    item_embeddings_url = st.secrets.files.item_embeddings_url
    item_embeddings_tarbz = "item_embeddings.tar.bz2"
    item_embeddings_folder = "data/"

    try:
        wget.download(item_embeddings_url, item_embeddings_tarbz)

        logging.debug("Extracting item embeddings folder.")
        with tarfile.open(item_embeddings_tarbz, "r:bz2") as tar:
            tar.extractall(item_embeddings_folder)

        os.remove(item_embeddings_tarbz)
        logging.debug("Item embeddings folder downloaded and extracted.")
    except Exception as e:
        logging.error(f"Error downloading item embeddings folder: {e}")


async def upload_conversation_logs_to_hf(
    conversation_log_file_path: str, repo_filename: str
) -> None:
    """Uploads conversation logs to Hugging Face asynchronously.

    Args:
        conversation_log_file_path: Path to the conversation log file locally.
        repo_filename: Name of the file in the Hugging Face repository.

    Raises:
        Exception: If an error occurs during the upload.
    """
    logging.debug(
        "Uploading conversation logs to Hugging Face: "
        f"{conversation_log_file_path}."
    )
    try:
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: HF_API.upload_file(
                path_or_fileobj=conversation_log_file_path,
                path_in_repo=repo_filename,
                repo_id=st.secrets.hf.dataset_repo,
                repo_type="dataset",
            ),
        )
        logging.debug("Conversation logs uploaded to Hugging Face.")
    except Exception as e:
        logging.error(
            f"Error uploading conversation logs to Hugging Face: {e}"
        )


async def upload_feedback_to_gsheet(
    row: Dict[str, str], worksheet: str = "votes"
) -> None:
    """Uploads feedback to Google Sheets asynchronously.

    Args:
        row: Row to upload to the worksheet.
        worksheet: Name of the worksheet to upload the feedback to.

    Raises:
        Exception: If an error occurs during the upload.
    """
    logging.debug("Uploading feedback to Google Sheets.")
    try:
        gs_connection = st.connection("gsheets", type=GSheetsConnection)
        df = gs_connection.read(worksheet=worksheet)
        if df[df["id"] == row["id"]].empty:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            # Add feedback to existing row
            df.loc[df["id"] == row["id"], "feedback"] = row["feedback"]
        gs_connection.update(data=df, worksheet=worksheet)
        logging.debug("Feedback uploaded to Google Sheets.")
    except Exception as e:
        logging.error(f"Error uploading feedback to Google Sheets: {e}")
