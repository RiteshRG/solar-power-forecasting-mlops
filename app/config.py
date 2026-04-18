import os

# ── MLflow ─────────────────────────────────────────────────────────────────
MODEL_PATH = "model"

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
    "15 minutes": 900,
    "5 minutes": 300,
    "1 minute":  60,
}

# ── Weather API ────────────────────────────────────────────────────────────
WEATHER_API_KEY   = os.getenv("WEATHER_API_KEY")          # never hardcode
CITY              = os.getenv("SOLAR_CITY", "Mumbai")
WEATHER_CACHE_TTL = 300   # seconds – how often to re-fetch
