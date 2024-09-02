"""Utility functions for CRS Arena."""

import logging
import os
import sqlite3
from datetime import timedelta
from typing import Any, Dict, List

import streamlit as st
import yaml

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
        model_args["api_key"] = st.secrets.openai.api_key

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
