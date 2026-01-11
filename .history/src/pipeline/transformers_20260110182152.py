from sklearn.compose import ColumnTransformer
from build_pipeline import num_pipeline
from build_pipeline import cat_pipeline
from features import num_cols
from features import cat_cols


col_transformer = ColumnTransformer(transformers=[
    ('num_pipeline', num_pipeline, num_cols),
    ('cat_pipeline', cat_pipeline, cat_cols)
], remainder='drop', n_jobs=-1)
