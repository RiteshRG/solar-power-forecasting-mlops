"""
SolarPredict – FastAPI Prediction API
======================================
Drop this file into the same `app/` folder alongside:
  config.py | model_loader.py | utils.py | weather_service.py

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

All existing files are untouched – this file only imports from them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Re-use existing project modules (zero changes to them) ─────────────────
from config import CSV_PATH, PLANT_CAPACITY_KW, WEATHER_API_KEY, CITY, WEATHER_CACHE_TTL
from model_loader import load_model
from utils import create_features, load_csv, save_prediction
from weather_service import get_weather_data

# ═══════════════════════════════════════════════════════════════════════════
# APP + MODEL INIT
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="SolarPredict API",
    description="FastAPI wrapper around the SolarPredict ML model.",
    version="1.0.0",
)

# Load model once at startup (same as Streamlit does)
_model = load_model()


# ═══════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    datetime_str: str = Field(
        ...,
        example="2024-06-15 13:30:00",
        description="Datetime in format YYYY-MM-DD HH:MM:SS",
    )
    irradiation: float = Field(..., ge=0.0, le=1.0, example=0.75)
    amb_temp: float    = Field(..., example=32.5)
    save: bool         = Field(True, description="Whether to save prediction to CSV")


class PredictResponse(BaseModel):
    ac_power_kw:       float
    capacity_pct:      float
    hour:              int
    hour_sin:          float
    hour_cos:          float
    irradiation:       float
    amb_temp:          float
    datetime_str:      str
    saved:             bool


class WeatherResponse(BaseModel):
    temperature:  float
    humidity:     int
    irradiation:  float
    clouds_pct:   int
    description:  str
    city:         str
    fetched_at:   str


class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool


# ═══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health():
    """Check API and model status."""
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PredictRequest):
    """
    Run a single manual prediction.

    - Accepts datetime, irradiation, and ambient temperature.
    - Uses the same `create_features()` logic as the Streamlit app.
    - Optionally saves the result to predictions.csv (save=true by default).
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # Reuse existing feature engineering (unchanged)
    try:
        feat_df, inter = create_features(req.datetime_str, req.irradiation, req.amb_temp)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    prediction = round(float(_model.predict(feat_df)[0]), 2)
    capacity_pct = round(prediction / PLANT_CAPACITY_KW * 100, 2)

    saved = False
    if req.save:
        save_prediction(
            CSV_PATH,
            req.datetime_str,
            req.irradiation,
            req.amb_temp,
            inter["hour_sin"],
            inter["hour_cos"],
            prediction,
            mode="api",
        )
        saved = True

    return PredictResponse(
        ac_power_kw=prediction,
        capacity_pct=capacity_pct,
        hour=inter["hour"],
        hour_sin=inter["hour_sin"],
        hour_cos=inter["hour_cos"],
        irradiation=req.irradiation,
        amb_temp=req.amb_temp,
        datetime_str=req.datetime_str,
        saved=saved,
    )


@app.post("/predict/now", response_model=PredictResponse, tags=["Prediction"])
def predict_now(save: bool = True):
    """
    Run a prediction for the current moment using live weather data.

    - If WEATHER_API_KEY is set, fetches live irradiation & temperature.
    - Falls back to last CSV row, then random values — same priority as the
      Streamlit Real-Time Monitor's auto-prediction logic.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    import random

    df = load_csv(CSV_PATH)

    weather = get_weather_data(WEATHER_API_KEY, CITY) if WEATHER_API_KEY else None
    if weather:
        sim_irr  = weather["irradiation"]
        sim_temp = weather["temperature"]
    elif len(df) > 0:
        last     = df.iloc[-1]
        sim_irr  = float(last["irradiation"])
        sim_temp = float(last["amb_temp"])
    else:
        sim_irr  = round(random.uniform(0.3, 0.9), 3)
        sim_temp = round(random.uniform(20.0, 35.0), 1)

    sim_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    feat_df, inter = create_features(sim_dt, sim_irr, sim_temp)
    prediction     = round(float(_model.predict(feat_df)[0]), 2)
    capacity_pct   = round(prediction / PLANT_CAPACITY_KW * 100, 2)

    if save:
        save_prediction(
            CSV_PATH, sim_dt, sim_irr, sim_temp,
            inter["hour_sin"], inter["hour_cos"],
            prediction, mode="api-auto",
        )

    return PredictResponse(
        ac_power_kw=prediction,
        capacity_pct=capacity_pct,
        hour=inter["hour"],
        hour_sin=inter["hour_sin"],
        hour_cos=inter["hour_cos"],
        irradiation=sim_irr,
        amb_temp=sim_temp,
        datetime_str=sim_dt,
        saved=save,
    )


@app.get("/weather", response_model=WeatherResponse, tags=["Weather"])
def weather():
    """
    Fetch current weather via OpenWeatherMap (requires WEATHER_API_KEY env var).
    Uses the same `get_weather_data()` function as the Streamlit app.
    """
    if not WEATHER_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="WEATHER_API_KEY environment variable is not set.",
        )
    data = get_weather_data(WEATHER_API_KEY, CITY)
    if data is None:
        raise HTTPException(status_code=502, detail="Failed to fetch weather data.")
    return WeatherResponse(**data)


@app.get("/history", tags=["Data"])
def history(limit: Optional[int] = 50):
    """
    Return the last `limit` rows from predictions.csv.
    Uses the same `load_csv()` helper as the Streamlit app.
    """
    df = load_csv(CSV_PATH)
    if df.empty:
        return {"count": 0, "records": []}

    df_out = df.sort_values("created_at", ascending=False).head(limit)
    # Convert timestamps to strings for JSON serialisation
    for col in ["datetime", "created_at"]:
        if col in df_out.columns:
            df_out[col] = df_out[col].astype(str)

    return {"count": len(df_out), "records": df_out.to_dict(orient="records")}