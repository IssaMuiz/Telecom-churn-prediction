from sklearn.linear_model import LogisticRegression


def build_baseline_model(random_state=42):
    """
    Build and return a baseline Logistic Regression model.

    Parameters:
    random_state (int): Seed for random number generator for reproducibility.

    Returns:
    LogisticRegression: An instance of Logistic Regression model.
    """
    model = LogisticRegression(random_state=random_state)
    return model


def train_baseline_model(model, X_train, y_train):
    """
    Train the provided Logistic Regression model on the training data.

    Parameters:
    model (LogisticRegression): The Logistic Regression model to be trained.
    X_train (array-like): Training feature data.
    y_train (array-like): Training target labels.

    Returns:
    LogisticRegression: The trained Logistic Regression model.
    """
    model.fit(X_train, y_train)
    return model
