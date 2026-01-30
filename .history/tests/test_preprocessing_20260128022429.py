import sys
import pandas as pd
import numpy as np
import pytest
from src.data_exploratory.clean_data import drop_unused_columns, replace_empty_string, duplicated_rows, change_to_numeric

sys.path.append('..')


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


def test_drop_unused_columns(raw_data):
    """
    Docstring for test_drop_unused_columns

    :param raw_data: Description
    :type raw_data: DataFrame
    """
    columns_to_drop = ["ID", "Country"]
    cleaned_data = drop_unused_columns(raw_data, columns_to_drop)
    assert "ID" not in cleaned_data.columns and "Country" not in cleaned_data.columns


def test_replace_empty_string(raw_data):
    """
    Docstring for test_replace_empty_string

    :param raw_data: Description
    :type raw_data: DataFrame
    """
    cleaned_data = replace_empty_string(raw_data)
    assert cleaned_data['gender'].isna().sum() == 2


def test_check_duplicated_rows(raw_data):
    """
    Docstring for test_check_duplicated_rows

    :param raw_data: Description
    :type raw_data: DataFrame
    """

    duplicated_count = duplicated_rows(raw_data)
    assert duplicated_count >= 1


def test_change_to_numeric(raw_data):
    """
    Docstring for test_change_to_numeric

    :param raw_data: Description
    :type raw_data: DataFrame
    """

    cleaned_data = change_to_numeric(raw_data, "TotalCharges")
    assert cleaned_data.loc[0, 1] == 1200.0
