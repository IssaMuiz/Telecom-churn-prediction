from sklearn.impute import SimpleImputer
import pandas as pd


def impute_missing_values(df: pd.DataFrame, col: str):
    """Impute missing values in specified columns of a pandas DataFrame using the median strategy.
    """
    impute = SimpleImputer(strategy='median')
    df[col] = impute.fit_transform(df[col])
