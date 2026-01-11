from sklearn.compose import ColumnTransformer
from src.pipeline.build_pipeline import num_pipeline
from src.pipeline.build_pipeline import cat_pipeline
from src.pipeline.features import num_cols
from src.pipeline.features import cat_cols


col_transformer = ColumnTransformer(transformers=[
    ('num_pipeline', num_pipeline, num_cols),
    ('cat_pipeline', cat_pipeline, cat_cols)
], remainder='drop', n_jobs=-1)
