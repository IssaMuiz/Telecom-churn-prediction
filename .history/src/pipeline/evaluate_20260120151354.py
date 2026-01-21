from sklearn.metrics import roc_auc_score, classification_report
import json


def evaluate_model(model, X_val, y_val, threshold=0.3):
    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "auc_roc": roc_auc_score(y_val, y_proba),
        "classification_report": classification_report(y_val, y_pred, output_dict=True),
        "threshold": threshold
    }

    return metrics
