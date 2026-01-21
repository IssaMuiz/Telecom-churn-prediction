from sklearn.metrics import roc_auc_score, classification_report


def evaluate_model(model, X_val, y_val, threshold=0.3):
    """
    Docstring for evaluate_model

    :param model: Description
    :param X_val: Description
    :param y_val: Description
    :param threshold: Description
    """
    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "auc_roc": roc_auc_score(y_val, y_proba),
        "classification_report": classification_report(y_val, y_pred, output_dict=True),
        "threshold": threshold
    }

    return metrics
