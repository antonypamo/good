import json
import os
from typing import Dict, List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

MODEL_PATH = os.getenv("MODEL_PATH", "logreg_rrf_savant.joblib")
MODEL_CONFIG_PATH = os.getenv("MODEL_CONFIG_PATH", "config.json")


def _load_expected_dim(config_path: str = MODEL_CONFIG_PATH, default_dim: int = 15) -> int:
    """Return the expected feature dimension from the model config if available."""

    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        return int(config.get("input_features", {}).get("dimension", default_dim))
    except FileNotFoundError:
        return default_dim
    except (json.JSONDecodeError, TypeError, ValueError):
        return default_dim


EXPECTED_DIM = _load_expected_dim()

app = FastAPI(title="RRF Savant Model API", version="1.0.0")


class Features(BaseModel):
    features: List[float] = Field(
        ..., description="Ordered feature vector expected by the RRF Savant logistic regression model"
    )

    @field_validator("features")
    @classmethod
    def validate_length(cls, value: List[float]) -> List[float]:
        expected_dim = EXPECTED_DIM
        if len(value) != expected_dim:
            raise ValueError(f"Expected {expected_dim} features, received {len(value)}")
        return value


class Prediction(BaseModel):
    label: str
    probabilities: Dict[str, float]


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
