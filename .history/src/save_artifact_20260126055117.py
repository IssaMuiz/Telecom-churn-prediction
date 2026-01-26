import os
import joblib
import json


def save_model(model, version='v2'):
    """
    Docstring for save_model

    :param model: Description
    :param version: Description
    """
    os.mkdir('artifacts/models/')
    path = f'artifacts/models/churn_model_{version}.pkl'
    joblib.dump(model, path)
    return path


def save_metrics(metrics, version='v2'):
    """
    Docstring for save_metrics

    :param metrics: Description
    :param version: Description
    """
    os.mkdir('artifacts/metrics/')
    path = f'artifacts/metrics/metrics_{version}.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    return path


def save_config(config, version='v2'):
    """
    Docstring for save_config

    :param config: Description
    :param version: Description
    """
    os.mkdir('artifacts/configs/')
    path = f'artifacts/configs/config_{version}.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)
    return path
