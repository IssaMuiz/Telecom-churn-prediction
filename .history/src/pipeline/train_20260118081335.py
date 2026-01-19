from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from src.pipeline.transformers import col_transformer
from src.pipeline.features import add_engineered_features


def run_pipeline(X_train, y_train):
    """
    Build and train a machine learning pipeline with preprocessing and Logistic Regression.
    Parameters:
    X_train (array-like): Training feature data.
    y_train (array-like): Training target labels.
    """
    X_train_fe = add_engineered_features(
        X_train)  # Apply all engineered features

    lr = LogisticRegression()

    pipeline = make_pipeline(col_transformer, lr)
    pipeline.fit(X_train_fe, y_train)
    return pipeline
