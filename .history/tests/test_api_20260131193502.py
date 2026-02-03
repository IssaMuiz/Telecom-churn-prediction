import sys
from fastapi.testclient import TestClient
from src.api import app

sys.path.append('..')

client = TestClient(app)


def test_predict_churn():
    """
    Test the /predict endpoint of the API.
    """
    sample_input = {
        'Gender': 'Male',
        'SeniorCitizen': 'No',
        'Partner': 'Yes',
        'Dependents': 'No',
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'DSL',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'TenureMonths': 1.0,
        'MonthlyCharges': 29.85,
        'TotalCharges': 29.85
    }

    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    assert "churn_prediction" in response.json()
    assert "churn_probability" in response.json()
