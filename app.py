import os
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

MODEL_PATH = os.getenv("MODEL_PATH", "logreg_rrf_savant.joblib")

app = FastAPI(title="RRF Savant Model API", version="1.0.0")


class Features(BaseModel):
    features: List[float] = Field(
        ..., description="Ordered feature vector expected by the RRF Savant logistic regression model"
    )

    @field_validator("features")
    @classmethod
    def validate_length(cls, value: List[float]) -> List[float]:
        expected_dim = 15
        if len(value) != expected_dim:
            raise ValueError(f"Expected {expected_dim} features, received {len(value)}")
        return value


class Prediction(BaseModel):
    label: str
    probabilities: dict


@app.on_event("startup")
def load_model() -> None:
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model artifact not found at '{MODEL_PATH}'. Set MODEL_PATH to the correct file before starting the API."
        )
    app.state.model = joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(payload: Features) -> Prediction:
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=500, detail="Model not loaded")

    model = app.state.model
    vector = np.array(payload.features, dtype=float).reshape(1, -1)

    try:
        proba = model.predict_proba(vector)[0]
        classes = getattr(model, "classes_", [])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {exc}") from exc

    probabilities = {str(label): float(score) for label, score in zip(classes, proba)}
    label_index = int(np.argmax(proba))
    predicted_label = str(classes[label_index]) if len(classes) else "unknown"

    return Prediction(label=predicted_label, probabilities=probabilities)
