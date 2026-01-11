from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from transformers import col_transformer


def run_pipeline(X_train, y_train):
    """
    X_train: for dependent features
    y_train: for target feature

    """

    lr = LogisticRegression()

    pipefinal = make_pipeline(col_transformer, lr)
    pipefinal.fit(X_train, y_train)

    return pipefinal
