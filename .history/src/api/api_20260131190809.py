import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.inference import make_inference

sys.path.append('..')

app = FastAPI(title="Customer Churn Prediction API")


class CustomerInput(BaseModel):
    """Pydantic model for customer input data."""

    Gender: str
    SeniorCitizen: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    TenureMonths: float
    MonthlyCharges: float
    TotalCharges: float


@app.post("/predict")
def predict_churn(customer: CustomerInput):
    """
    API endpoint to predict customer churn.
    params: customer: CustomerInput: Input data for prediction
    return: prediction and probability
    """
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([customer.model_dump()])

        # Make inference
        prediction, probability = make_inference(input_data)

        # Return the result
        return {
            "churn_prediction": int(prediction[0]),
            "churn_probability": float(probability[0])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
