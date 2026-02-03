import pandas as pd
from src.model_registry import load_model


MODEL_NAME = 'churn_model_v3'


def make_inference(data: pd.DataFrame):
    """
    Make predictions using the trained model.
    params: data: pd.DataFrame: Input data for prediction
    return: predictions and probabilities
    """
    model = load_model(MODEL_NAME)
    predictions = model.predict(data)
    # Probability of the positive class
    probabilities = model.predict_proba(data)[:, 1]
    return predictions, probabilities
