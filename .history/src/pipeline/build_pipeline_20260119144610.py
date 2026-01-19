from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


# Define pipelines for numerical and categorical features
num_pipeline = Pipeline(steps=[
    # Impute missing values with median
    ('imputer', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())  # Standardize numerical features
])


cat_pipeline = Pipeline(steps=[
    # One-hot encode categorical features
    ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
