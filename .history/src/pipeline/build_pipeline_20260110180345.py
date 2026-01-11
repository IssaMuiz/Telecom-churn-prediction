from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])


cat_pipeline = Pipeline(steps=[
    ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output='false'))
])
