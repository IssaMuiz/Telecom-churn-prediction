from sklearn.impute import SimpleImputer
import pandas as pd


def impute_missing_value(df: pd.DataFrame, col: [str]):
    impute = SimpleImputer(strategy='median')
    df[col] = impute.fit_transform(df[col])
