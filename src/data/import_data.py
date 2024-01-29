import os
import pandas as pd  # type: ignore
import yaml
import logging
import requests
import io


def import_yaml_config(location: str) -> dict:
    """Wrapper to easily import YAML

    Args:
        location (str): File path

    Returns:
        dict: YAML content as dict
    """
    if not isinstance(location, str):
        raise TypeError("File path must be a string.")

    if not os.path.isfile(location):
        raise FileNotFoundError(f"Invalid file path: {location}")

    try:
        with open(location, "r") as f:
            dict_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logging.error(f"Error loading YAML file: {e}")
        dict_config = {}

    return dict_config


def import_data(path: str) -> pd.DataFrame:
    """Import Titanic datasets

    Args:
        path (str): File location

    Returns:
        pd.DataFrame: Titanic dataset
    """
    if os.path.exists(path):
        data = pd.read_csv(path)
    elif path.startswith("https://"):
        response = requests.get(path)
        response.raise_for_status()
        data = pd.read_csv(io.StringIO(response.text))
    else:
        raise FileNotFoundError("File does not exist at the given path.")
    data = data.drop(columns="PassengerId")
    return data
