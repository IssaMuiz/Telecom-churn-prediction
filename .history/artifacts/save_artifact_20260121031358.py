import os
import joblib
import json
from datetime import datetime


def save_model(model, version='v1'):
    """
    Docstring for save_model

    :param model: Description
    :param version: Description
    """
    os.mkdir('artifacts/models/', exist_ok=True)
    path = f'artifacts/models/churn_model_{version}.pkl'
    joblib.dump(model, path)
