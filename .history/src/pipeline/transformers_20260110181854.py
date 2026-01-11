from sklearn.compose import ColumnTransformer
from .build_pipeline import num_pipeline
from pipeline.build_pipeline import cat_pipeline
from pipeline.features import num_cols
from pipeline.features import cat_cols


col_transformer = ColumnTransformer(transformers=[
    ('num_pipeline', num_pipeline, num_cols),
    ('cat_pipeline', cat_pipeline, cat_cols)
])
