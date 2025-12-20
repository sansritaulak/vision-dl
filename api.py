# api.py — FINAL VERSION THAT FIXES THE SWAGGER UI
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from inference import predict
import shutil
import os
from pydantic import BaseModel
from typing import Dict

app = FastAPI(title="CIFAR-10 Classifier (Calibrated)", version="1.0")

# THIS FIXES THE RED ERRORS IN /docs
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    all_probabilities: Dict[str, float]

@app.post("/predict", response_model=PredictionResponse)
async def predict_api(file: UploadFile = File(...)):
    os.makedirs("temp", exist_ok=True)
    path = f"temp/{file.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    try:
        result = predict(path)
        return result  # ← can just return the dict now!
    finally:
        if os.path.exists(path):
            os.remove(path)