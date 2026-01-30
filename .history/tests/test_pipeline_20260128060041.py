import sys
import pandas as pd
import numpy as np
import pytest
from src.pipeline.train import run_pipeline
from src.data_exploratory.split_data import split_features_target

sys.path.append("..")


@pytest.fixture
def raw_data():
    """
    Docstring for raw_data
    Provides a sample raw DataFrame for testing.
    Returns: pd.DataFrame
    """
    return pd.DataFrame({
        "ID": [1, 2, 3, 3],
        "gender": ["Male", "Female", " ", " "],
        "Country": ["USA", "Canada", "USA", "USA"],
        "SeniorCitizen": [0, 1, 0, 0],
        "MonthlyCharges": [70.5, 89.1, np.nan, np.nan],
        "TotalCharges": ['1200.0', '3400.0', '2300.0', '2300.0'],
        "Contract": ["Month-to-month", "Two year", "One year", "One year"],
        "Churn Value": [1, 0, 1, 1]
    })


@pytest.fixture
def split_data(raw_data):
    """
    Docstring for data splitting
    """
    X, y = split_features_target(raw_data)
    return X, y
