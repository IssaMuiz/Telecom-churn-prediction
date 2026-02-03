from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.inference import make_inference


app = FastAPI(title="Customer Churn Prediction API")


class CustomerInput(BaseModel):
    """Pydantic model for customer input data."""

    'Gender': str,
    'Senior Citizen': str,
    'Partner': str,
    'Dependents': str,
    'Phone Service': str,
    'Multiple Lines': str,
    'Internet Service': str,
    'Online Security': str,
    'Online Backup': str,
    'Device Protection': str,
    'Tech Support': str,
    'Streaming TV': str,
    'Streaming Movies': str,
    'Contract': str,
    'Paperless Billing': str,
    'Payment Method': str,
    'Tenure Months': float,
    'Monthly Charges': float,
    'Total Charges': float,
