from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.data_exploratory.evaluate import evaluate_model


def build_model(random_state=42):
    """
    Build and return a baseline Logistic Regression model.

    Parameters:
    cv (StratifiedKFold): Cross-validation strategy.
    random_state (int): Seed for random number generator for reproducibility.

    Returns:
    LogisticRegression: An instance of Logistic Regression model.
    """
    model = LogisticRegression(max_iter=10000,
                               solver="lbfgs", random_state=random_state)
    return model


def train_model(model, X_train, y_train):
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


def run_model(X_train, y_train, X_val, y_val):
    """
    Build, train, and evaluate a baseline Logistic Regression model.

    Parameters:
    X_train (array-like): Training feature data.
    y_train (array-like): Training target labels.
    X_val (array-like): Validation feature data.
    y_val (array-like): Validation target labels.

    Returns:
    dict: A dictionary containing evaluation metrics of the trained model.
    """
    model = build_model()
    trained_model = train_model(model, X_train, y_train)

    evaluation_metrics = evaluate_model(trained_model, X_val, y_val)

    return evaluation_metrics
