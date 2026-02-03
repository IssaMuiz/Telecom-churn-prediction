from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.inference import make_inference
