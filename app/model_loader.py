from __future__ import annotations

import mlflow
import mlflow.pyfunc

from config import MLFLOW_TRACKING_URI, MODEL_URI


def load_model():
    """
    Try loading the model from MLflow Model Registry first.
    Falls back to a local model.pkl if MLflow is unavailable.
    Returns None if both fail – the app handles None gracefully.
    """
    # ── 1. MLflow registry ─────────────────────────────────────────────────
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model = mlflow.pyfunc.load_model(MODEL_URI)
        print(f"[model_loader] Loaded from MLflow: {MODEL_URI}")
        return model
    except Exception as mlflow_err:
        print(f"[model_loader] MLflow failed ({mlflow_err}), trying local fallback…")

    # ── 2. Local model.pkl fallback ────────────────────────────────────────
    try:
        import joblib
        model = joblib.load("model.pkl")
        print("[model_loader] Loaded from local model.pkl")
        return model
    except Exception as pkl_err:
        print(f"[model_loader] Local fallback also failed ({pkl_err})")

    return None
