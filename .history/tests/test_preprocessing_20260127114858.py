import sys
import pandas as pd
import numpy as np
import pytest
from src.data_exploratory.clean_data import drop_unused_columns

sys.path.append('..')


@pytest.fixture
def raw_data():
    """
    Docstring for raw_data
    Provides a sample raw DataFrame for testing.
    Returns: pd.DataFrame
    """
    return pd.DataFrame({
        "ID": [1, 2, 3],
        "gender": ["Male", "Female", "Female"],
        "Country": ["USA", "Canada", "USA"],
        "SeniorCitizen": [0, 1, 0],
        "MonthlyCharges": [70.5, 89.1, np.nan],
        "TotalCharges": [1200.0, 3400.0, 2300.0],
        "Contract": ["Month-to-month", "Two year", "One year"],
        "Churn Value": [1, 0, 1]
    })


def test_drop_unused_columns(raw_data):
    """
    Docstring for test_drop_unused_columns

    :param raw_data: Description
    :type raw_data: DataFrame
    """
    columns_to_drop = ["ID", "Country"]
    cleaned_data = drop_unused_columns(raw_data, columns_to_drop)
    assert "ID" not in cleaned_data.columns and "Country" not in cleaned_data.columns
