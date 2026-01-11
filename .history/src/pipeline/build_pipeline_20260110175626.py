from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SimpleImputer
from sklearn.preprocessing import StandardScalar


num_pipeline = Pipeline(steps=[
    ('imputer', simpleImputer(strategy='median')),
    ('scale', StandardScalar())
])
