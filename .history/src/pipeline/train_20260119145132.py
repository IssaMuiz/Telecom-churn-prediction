from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from src.pipeline.transformers import pipeline_preprocessing

# Build and train a machine learning pipeline with preprocessing and Logistic Regression.


def run_pipeline(X_train, y_train):
    """
    Build and train a machine learning pipeline with preprocessing and Logistic Regression.
    Parameters:
    X_train (array-like): Training feature data.
    y_train (array-like): Training target labels.
    """

    lr = LogisticRegression()  # Initialize Logistic Regression model

    # Create a pipeline with preprocessing and model
    pipeline = make_pipeline(pipeline_preprocessing, lr)
    pipeline.fit(X_train, y_train)  # Train the pipeline

    return pipeline  # Return the trained pipeline
