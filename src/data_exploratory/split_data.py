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


def split_features_target(df=pd.DataFrame):
    """Split the DataFrame into features and target variable.

    Args:
        df (pd.DataFrame): The DataFrame to split.
    """

    X = df.drop(columns=['Churn Value'])
    y = df['Churn Value']

    return X, y
