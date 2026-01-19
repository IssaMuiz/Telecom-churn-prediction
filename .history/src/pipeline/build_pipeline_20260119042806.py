from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from src.pipeline.features import add_engineered_features


num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])


cat_pipeline = Pipeline(steps=[
    ('add_engineered_features', add_engineered_feaures())
    ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
