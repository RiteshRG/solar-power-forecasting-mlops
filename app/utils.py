from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd


from config import CSV_COLUMNS


# ── Feature engineering ────────────────────────────────────────────────────

def create_features(
    datetime_str: str,
    irradiation: float,
    amb_temp: float,
) -> tuple[pd.DataFrame, dict]:
    """
    Parse inputs, compute cyclical hour features, and return:
      - feature_df  : pd.DataFrame ready for model.predict()
      - intermediates: dict with hour / hour_sin / hour_cos for display
    Raises ValueError with a human-readable message on bad datetime.
    """
    try:
        dt = datetime.strptime(datetime_str.strip(), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        raise ValueError(
            f"Cannot parse '{datetime_str}'. "
            "Expected format: YYYY-MM-DD HH:MM:SS"
        )

    hour     = dt.hour
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    feature_df = pd.DataFrame(
        [[irradiation, amb_temp, hour_sin, hour_cos]],
        columns=["IRRADIATION", "AMBIENT_TEMPERATURE", "HOUR_SIN", "HOUR_COS"],
    )

    intermediates = {
        "hour":     hour,
        "hour_sin": round(hour_sin, 4),
        "hour_cos": round(hour_cos, 4),
    }
    return feature_df, intermediates


# ── CSV helpers ────────────────────────────────────────────────────────────

def load_csv(csv_path: str) -> pd.DataFrame:
    """
    Load the predictions CSV.
    Creates the file with headers if it does not exist.
    Returns an empty DataFrame (never raises) on any read error.
    """
    if not os.path.exists(csv_path):
        empty = pd.DataFrame(columns=CSV_COLUMNS)
        try:
            empty.to_csv(csv_path, index=False)
        except Exception:
            pass
        return empty

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return pd.DataFrame(columns=CSV_COLUMNS)

        # Ensure expected columns exist
        for col in CSV_COLUMNS:
            if col not in df.columns:
                df[col] = None

        df["datetime"]   = pd.to_datetime(df["datetime"],   errors="coerce")
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        return df

    except Exception:
        return pd.DataFrame(columns=CSV_COLUMNS)


def save_prediction(
    csv_path: str,
    datetime_str: str,
    irradiation: float,
    amb_temp: float,
    hour_sin: float,
    hour_cos: float,
    ac_power: float,
    mode: str,
) -> None:
    """
    Append one prediction row to the CSV.
    Wrapped in try/except – never crashes the app.
    """
    try:
        existing = load_csv(csv_path)
        next_id  = int(existing["id"].max()) + 1 if len(existing) > 0 else 1

        new_row = pd.DataFrame([{
            "id":          next_id,
            "datetime":    datetime_str,
            "irradiation": round(irradiation, 4),
            "amb_temp":    round(amb_temp, 2),
            "hour_sin":    round(hour_sin, 4),
            "hour_cos":    round(hour_cos, 4),
            "ac_power":    round(ac_power, 2),
            "mode":        mode,
            "created_at":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }])

        # ── CloudWatch log ─────────────────────────────────────────────
        print(f"[PREDICTION] id={next_id} | datetime={datetime_str} | "
              f"irradiation={irradiation} | amb_temp={amb_temp} | "
              f"ac_power={round(ac_power, 2)} kW | mode={mode}")

        updated = pd.concat([existing, new_row], ignore_index=True)
        updated.to_csv(csv_path, index=False)

    except Exception as e:
        print(f"⚠️ Could not save prediction: {e}")
