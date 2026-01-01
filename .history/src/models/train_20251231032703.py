from sklearn.linear_model import LogisticRegression


def baseline_model(X, y):
    """Train a baseline logistic regression model.

    Args:
        X (pd.DataFrame): Feature set.
        y (pd.Series): Target variable.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model
