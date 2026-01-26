from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from src.pipeline.transformers import pipeline_preprocessing
from sklearn.model_selection import GridSearchCV

# Build and train a machine learning pipeline with preprocessing and Logistic Regression.


def run_pipeline(X_train, y_train):
    """
    Build and train a machine learning pipeline with preprocessing and Logistic Regression.
    Parameters:
    X_train (array-like): Training feature data.
    y_train (array-like): Training target labels.
    """

    # Initialize Logistic Regression model
    lr = LogisticRegression(random_state=45)

    param_grid = [{
        'C': [0.01, 0.1, 1.0, 10,],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'class_weight': [None, 'balanced']
    }]

    lr_grid = GridSearchCV(
        estimator=lr,
        param_grid=param_grid,
        scoring=['recall'],
        refit='recall',
        cv=5,
        n_jobs=-1,
        error_score='raise'
    )

    lr_grid.fit(X_train, y_train)

    best_estimator = lr_grid.best_estimator_

    # Create a pipeline with preprocessing and model
    pipeline = make_pipeline(pipeline_preprocessing, lr)
    pipeline.fit(X_train, y_train)  # Train the pipeline

    return pipeline  # Return the trained pipeline
