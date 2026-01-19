from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from src.pipeline.transformers import col_transformer
from src.features.features_construction import no_family


def run_pipeline(X_train, y_train):
    """
    Build and train a machine learning pipeline with preprocessing and Logistic Regression.
    Parameters:
    X_train (array-like): Training feature data.
    y_train (array-like): Training target labels.
    """
    X_train_fe = no_family(X_train)  # Apply the no_family feature engineering

    lr = LogisticRegression()

    pipeline = make_pipeline(col_transformer, lr)
    pipeline.fit(X_train_fe, y_train)
    return pipeline
