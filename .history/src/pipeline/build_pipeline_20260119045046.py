from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from src.pipeline.features import add_engineered_features
import pandas as pd


num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])


cat_pipeline = Pipeline(steps=[
    ('add_engineered_features', add_engineered_features(df: pd.DataFrame)),
    ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
