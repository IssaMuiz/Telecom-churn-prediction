from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.inference import make_inference


app = FastAPI(title="Customer Churn Prediction API")


class CustomerInput(BaseModel):

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
