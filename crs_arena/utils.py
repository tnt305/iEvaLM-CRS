"""Utility functions for CRS Arena."""

import logging
import os
import sqlite3
import sys
import tarfile
from datetime import timedelta
from typing import Any, Dict, List

import openai
import streamlit as st
import wget
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model.crs_model import CRSModel


@st.cache_resource(show_spinner="Loading CRS...", ttl=timedelta(days=1))
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
    wget.download(models_url, models_targz)

    logging.debug("Extracting models folder.")
    with tarfile.open(models_targz, "r:gz") as tar:
        tar.extractall(models_folder)

    os.remove(models_targz)
    logging.debug("Models folder downloaded and extracted.")


def download_and_extract_item_embeddings() -> None:
    """Downloads the item embeddings folder from the server and extracts it."""
    logging.debug("Downloading item embeddings folder.")
    item_embeddings_url = st.secrets.files.item_embeddings_url
    item_embeddings_tarbz = "item_embeddings.tar.bz2"
    item_embeddings_folder = "data/"
    wget.download(item_embeddings_url, item_embeddings_tarbz)

    logging.debug("Extracting item embeddings folder.")
    with tarfile.open(item_embeddings_tarbz, "r:bz2") as tar:
        tar.extractall(item_embeddings_folder)

    os.remove(item_embeddings_tarbz)
    logging.debug("Item embeddings folder downloaded and extracted.")
