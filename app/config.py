import os

# ── MLflow ─────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_URI           = "models:/SolarPowerModel/Production"

# ── Storage ────────────────────────────────────────────────────────────────
CSV_PATH          = "predictions.csv"
PLANT_CAPACITY_KW = 5000

CSV_COLUMNS = [
    "id", "datetime", "irradiation", "amb_temp",
    "hour_sin", "hour_cos", "ac_power", "mode", "created_at"
]

# ── Scheduler ──────────────────────────────────────────────────────────────
INTERVAL_OPTIONS = {
    "30 minutes": 1800,
    "1 hour":     3600,
    "2 hours":    7200,
}

# ── Weather API ────────────────────────────────────────────────────────────
WEATHER_API_KEY   = os.getenv("WEATHER_API_KEY")          # never hardcode
CITY              = os.getenv("SOLAR_CITY", "Mumbai")
WEATHER_CACHE_TTL = 300   # seconds – how often to re-fetch
