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
        "gender": ["Male", "Female"] * 6,
        "SeniorCitizen": [0, 1] * 6,
        "MonthlyCharges": [50, 60, 70, 80, 90, 100, 55, 65, 75, 85, 95, 105],
        "TotalCharges": ["1000", "2000", "3000", "4000", "5000", "6000",
                         "1100", "2100", "3100", "4100", "5100", "6100"],
        "Contract": ["Month-to-month", "One year", "Two year"] * 4,
        "Churn": [0, 1] * 6
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
