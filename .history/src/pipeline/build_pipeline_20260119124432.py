import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from src.pipeline.features import add_engineered_features

feature_engineering = FunctionTransformer(
    add_engineered_features, validate=True)


num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])


cat_pipeline = Pipeline(steps=[
    ('feature_engineering', feature_engineering),
    ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
