from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

impute = SimpleImputer(strategy='median')
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
scaler = StandardScaler()


def impute_missing_values(df: pd.DataFrame, col: list):
    """Impute missing values in specified columns of a pandas DataFrame using the median strategy.
    """
    df[col] = impute.fit_transform(df[col])


def onehotencoding(df: pd.DataFrame):
    """Apply one-hot encoding to categorical columns in a pandas DataFrame.
    """
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    df_encoded = pd.DataFrame(encoder.fit_transform(df[cat_cols]))

    df_encoded.index = df.index  # Align indices with the original DataFrame
    df_encoded.columns = encoder.get_feature_names_out(
        input_features=cat_cols)  # Get new column names

    df = df.drop(columns=cat_cols, axis=1)  # Drop original categorical columns
    df = pd.concat([df, df_encoded], axis=1)  # Concatenate the encoded columns

    return df


def standard_scaling(df: pd.DataFrame):
    """Apply standard scaling to specified numeric columns in a pandas DataFrame.
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    num_cols.remove('Churn Value')
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df[num_cols].head()


def impute_missing_values_for_non_train_set(df: pd.DataFrame, cols: list):
    """Impute missing values in specified columns of a pandas DataFrame using the median strategy.

    Args:
        df (pd.DataFrame): The DataFrame in which to impute missing values.
        cols (list): List of column names to impute missing values for.
    """

    df[cols] = impute.transform(df[cols])


def onehotencoding_for_non_train_set(df: pd.DataFrame):
    """Apply one-hot encoding to categorical columns in a pandas DataFrame.
    """
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    df_encoded = pd.DataFrame(encoder.transform(df[cat_cols]))

    df_encoded.index = df.index  # Align indices with the original DataFrame
    df_encoded.columns = encoder.get_feature_names_out(
        input_features=cat_cols)  # Get new column names

    df = df.drop(columns=cat_cols, axis=1)  # Drop original categorical columns
    df = pd.concat([df, df_encoded], axis=1)  # Concatenate the encoded columns

    return df


def standard_scaling_for_non_train_set(df: pd.DataFrame):
    """Apply standard scaling to specified numeric columns in a pandas DataFrame.
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    num_cols.remove('Churn Value')
    df[num_cols] = scaler.transform(df[num_cols])
    return df[num_cols].head()


def normalize_with_log(df: pd.DataFrame, column: list[str]):
    """ Apply log normalization to a list of columns in a pandas DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame in which to apply log normalization.
        column (list[str]): The column names to normalize.
    """
    for col in column:
        df[col + "_log"] = np.log1p(df[col])
        df.drop(columns=[col], inplace=True)
    return df.head()
