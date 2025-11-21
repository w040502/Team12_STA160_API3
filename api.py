# api.py
# FastAPI wrapper around the Airbnb price prediction model.

from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from airbnb_predict import predict_price, artifacts

app = FastAPI(
    title="Airbnb Price Prediction API",
    description="Per-city stacked ensemble (XGB + RF + Linear meta-model)",
    version="1.0.0",
)

# ------------------------------------------------------------
# CORS (so your HTML/JS frontend can call this API)
# ------------------------------------------------------------

# If you know your frontend origin (e.g., GitHub Pages), put it here.
# Example:
# origins = ["https://w040502.github.io"]
origins = ["*"]  # for now, allow all origins; you can tighten later.

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# Pydantic models for request/response
# ------------------------------------------------------------

class PredictRequest(BaseModel):
    city: str = Field(..., description="City name, e.g. 'San Francisco'")
    features: Dict[str, Any] = Field(
        default_factory=dict,
        description="Mapping of feature name to value. "
                    "Missing features will be filled with city-specific medians."
    )


class PredictResponse(BaseModel):
    city_key: str
    predicted_price: float


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------

@app.get("/", summary="Root")
def read_root():
    """Simple root endpoint."""
    return {
        "message": "Airbnb Price Prediction API",
        "available_cities": artifacts.get("cities", []),
    }


@app.get("/health", summary="Health check")
def health_check():
    """Health-check endpoint for Render."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse, summary="Predict Airbnb price")
def predict(req: PredictRequest):
    """
    Predict nightly price for a given city + feature set.
    """
    # `predict_price` will handle:
    # - normalizing the city name
    # - filling missing features with medians
    # - running stacked model and exp(log_price)
    price = predict_price(req.city, req.features)

    # Normalize city key just for returning (using artifacts)
    # If predict_price raised no error, city is valid.
    # We'll just echo the city as-is, or you can return the internal key.
    # For simplicity, we return the original `req.city`.
    return PredictResponse(
        city_key=req.city,
        predicted_price=price,
    )