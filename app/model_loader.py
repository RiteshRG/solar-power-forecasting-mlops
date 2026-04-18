from __future__ import annotations

import os
import joblib

from config import MODEL_PATH


def load_model():
    """
    Load ONLY from exported local model.
    No MLflow dependency at runtime (production safe).
    """

    model_file = os.path.join(MODEL_PATH, "model.pkl")

    try:
        model = joblib.load(model_file)
        print(f"[model_loader] Loaded local model: {model_file}")
        return model
    except Exception as e:
        print(f"[model_loader] Failed to load model: {e}")
        return None