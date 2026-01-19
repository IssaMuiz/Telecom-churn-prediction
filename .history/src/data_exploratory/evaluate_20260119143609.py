from sklearn.metrics import roc_auc_score, classification_report


def evaluate_baseline_model(model, X_val, y_val):
    """
    Evaluate the provided Logistic Regression model on the test data.

    Parameters:
    model (LogisticRegression): The trained Logistic Regression model to be evaluated.
    X_val (array-like): Test feature data.
    y_val (array-like): True labels for the test data.

    Returns:
    dict: A dictionary containing evaluation metrics including AUC-ROC and classification report.
    """
    y_proba = model.predict_proba(X_val)[:, 1]

    threshold = 0.3

    custom_y_pred = (y_proba >= threshold)

    auc_roc = roc_auc_score(y_val, y_proba)
    class_report = classification_report(
        y_val, custom_y_pred, output_dict=True)
    evaluation_metrics = {
        'AUC-ROC': auc_roc,
        'Classification Report': class_report
    }

    return evaluation_metrics
