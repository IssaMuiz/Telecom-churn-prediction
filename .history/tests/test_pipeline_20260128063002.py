import sys
import pandas as pd
import numpy as np
import pytest
from src.pipeline.train import run_pipeline
from src.data_exploratory.split_data import split_features_target, split_data

sys.path.append("..")


@pytest.fixture
def raw_data():
    """
    Docstring for raw_data
    Provides a sample raw DataFrame for testing.
    Returns: pd.DataFrame
    """
    return pd.DataFrame({
        "ID": [1, 2, 3, 3, 4, 5],
        "gender": ["Male", "Female", "Male", "Female", "Male", "Male"],
        "Country": ["USA", "Canada", "USA", "USA", "Canada", "USA"],
        "SeniorCitizen": [0, 1, 0, 0, 1, 2],
        "MonthlyCharges": [70.5, 89.1, np.nan, np.nan, 23.2, 45.2],
        "TotalCharges": ['1200.0', '3400.0', '2300.0', '2300.0', '4000.0', '2900.0'],
        "Contract": ["Month-to-month", "Two year", "One year", "One year", "One year", "Month-to-month"],
        "Churn Value": [1, 0, 1, 1, 0, 1]
    })


def test_run_pipeline(raw_data):
    """
    Docstring for running pipeline
    """
    X, y, z = split_data(raw_data)
    X_train, y_train = split_features_target(X)
    X_val, y_val = split_features_target(y)

    model = run_pipeline(X_train, y_train)
    score = model.score(X_val, y_val)
    assert score > 70
