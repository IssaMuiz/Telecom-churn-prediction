from sklearn.model_selection import train_test_split
import pandas as pd


def split_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    """Split the DataFrame into training and testing sets.

    Args:
        df (pd.DataFrame): The DataFrame to split.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
    """

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state)
    return train_df, test_df


def split_train_validation(df: pd.DataFrame, validation_size=0.25, random_state=42):
    """Split the training DataFrame into training and validation sets.

    Args:
        df (pd.DataFrame): The training DataFrame to split.
        validation_size (float): The proportion of the training dataset to include in the validation split.
        random_state (int): Random seed for reproducibility.
    """

    train_df, validation_df = train_test_split(
        df, test_size=validation_size, random_state=random_state)
    return train_df, validation_df
