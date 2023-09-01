import pandas as pd
import yaml


def import_yaml_config(file_path: str) -> dict:
    """Wrapper to easily import YAML

    Args:
        location (str): File path

    Returns:
        dict: YAML content as dict
    """

    with open(file_path, "r") as f:
        dict_config = yaml.safe_load(f)
    return dict_config


def import_data(path: str) -> pd.DataFrame:
    """Import Titanic datasets

    Args:
        path (str): File location

    Returns:
        pd.DataFrame: Titanic dataset
    """
    data = pd.read_csv(path)
    data = data.drop(columns="PassengerId")
    return data
