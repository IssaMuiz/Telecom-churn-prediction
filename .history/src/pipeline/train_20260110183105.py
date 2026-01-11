from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from transformers import col_transformer

lr = LogisticRegression()

pipefinal = make_pipeline(col_transformer, lr)
