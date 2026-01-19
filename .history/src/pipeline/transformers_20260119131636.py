from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from src.pipeline.build_pipeline import num_pipeline, cat_pipeline
from src.pipeline.features import num_cols, cat_cols, add_engineered_features


feature_engineering = FunctionTransformer(
    add_engineered_features, validate=False)

col_transformer = ColumnTransformer(transformers=[
    ('num_pipeline', num_pipeline, num_cols),
    ('cat_pipeline', cat_pipeline, cat_cols)
], remainder='drop', n_jobs=-1)
