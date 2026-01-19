from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from src.pipeline.transformers import col_transformer


def run_pipeline(X_train, y_train):
    """
    Build and train a machine learning pipeline with preprocessing and Logistic Regression.
    Parameters:
    X_train (array-like): Training feature data.
    y_train (array-like): Training target labels.
    """

    lr = LogisticRegression()

    pipeline = make_pipeline(col_transformer, lr)
    pipeline.fit(X_train, y_train)

    return pipeline
