from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from transformers import col_transformer


def run_pipeline(X_train, y_train):

    lr = LogisticRegression()

    pipefinal = make_pipeline(col_transformer, lr)
    pipefinal.fit(X_train, y_train)
