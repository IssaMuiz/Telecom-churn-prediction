from sklearn.model_selection import train_test_split
import pandas as pd


def split_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    """Split the DataFrame into training and testing sets.

    Args:
        df (pd.DataFrame): The DataFrame to split.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
    """

    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state)

    train_df, val_df = train_test_split(
        train_val_df, test_size=0.25, random_state=random_state)

    return train_df, val_df, test_df


def split_train_validation(df: pd.DataFrame, validation_size=0.25, random_state=42):
    """Split the training DataFrame into training and validation sets.

    Args:
        df (pd.DataFrame): The training DataFrame to split.
        validation_size (float): The proportion of the training dataset to include in the validation split.
        random_state (int): Random seed for reproducibility.
    """

    train_df, validation_df = train_test_split(
        df, test_size=validation_size, random_state=random_state)
    return train_df,


def split_train_val_test(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    val_size: float = 0.25,
    random_state: int = 42,
):
    """
    Split DataFrame into train, validation, and test sets.

    Args:
        df (pd.DataFrame): Full dataset
        target (str): Target column name
        test_size (float): Proportion for test set
        val_size (float): Proportion of train set used for validation
        random_state (int): Random seed

    Returns:
        train_df, val_df, test_df
    """

    # 1️⃣ Split out test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target]
    )

    # 2️⃣ Split train and validation
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=random_state,
        stratify=train_val_df[target]
    )

    return train_df, val_df, test_df
