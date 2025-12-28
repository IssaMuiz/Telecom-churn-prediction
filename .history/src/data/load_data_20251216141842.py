import pandas as pd
"""Importing the neccessary module"""
from pathlib import Path


def load_data(path: str | Path) -> pd.DataFrame:
    """Load data from an Excel file into a pandas DataFrame.

    Args:
        path (str | Path): The file path to the Excel file."""

    path = Path(path)  # Ensure the path is a Path object

    if not path.exists():
        # Check if the file exists
        raise FileNotFoundError(f"The file at {path} does not exist.")

    # Load the data into a DataFrame using pandas
    return pd.read_excel(path, engine='openpyxl')
