import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import os

# ---------------------------------------------------
# Load artifacts safely
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "arrhythmia_model.keras")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

try:
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    print("✅ Model and scaler loaded successfully")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load artifacts: {e}")

app = FastAPI(title="ECG Arrhythmia Detection API")


# ---------------------------------------------------
# Input Schema
# ---------------------------------------------------

class ECGInput(BaseModel):
    signal: list = Field(
        ...,
        min_items=187,
        max_items=187,
        description="ECG beat containing exactly 187 values"
    )


# ---------------------------------------------------
# Root Endpoint (optional but avoids 404 confusion)
# ---------------------------------------------------

@app.get("/")
def home():
    return {"message": "ECG Arrhythmia Detection API is running 🚀"}


# ---------------------------------------------------
# Prediction Endpoint
# ---------------------------------------------------

@app.post("/predict")
def predict(input_data: ECGInput):
    try:
        # Convert to numpy
        signal = np.array(input_data.signal).reshape(1, -1)

        # Normalize using training scaler
        signal = scaler.transform(signal)

        # Reshape for CNN
        signal = signal.reshape(1, signal.shape[1], 1)

        # Predict
        prediction = model.predict(signal)[0][0]

        label = "Abnormal" if prediction > 0.5 else "Normal"

        return {
            "prediction": label,
            "confidence": float(prediction)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
