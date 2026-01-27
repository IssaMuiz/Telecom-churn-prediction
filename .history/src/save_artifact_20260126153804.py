import os
import joblib
import json


def save_model(model, version='v3'):
    """
    Docstring for save_model

    :param model: Description
    :param version: Description
    """
    os.makedirs('artifacts/models/', exist_ok=True)  # Ensure directory exists
    # Define the file path
    path = f'artifacts/models/churn_model_{version}.pkl'
    joblib.dump(model, path)  # Save the model using joblib
    return path  # Return the path where the model is saved


def save_metrics(metrics, version='v3'):
    """
    Docstring for save_metrics

    :param metrics: Description
    :param version: Description
    """
    os.makedirs('artifacts/metrics/', exist_ok=True)
    path = f'artifacts/metrics/metrics_{version}.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    return path


def save_config(config, version='v3'):
    """
    Docstring for save_config

    :param config: Description
    :param version: Description
    """
    os.makedirs('artifacts/configs/', exist_ok=True)
    path = f'artifacts/configs/config_{version}.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)
    return path
