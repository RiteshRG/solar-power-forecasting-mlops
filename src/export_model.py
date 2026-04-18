import mlflow
import mlflow.pyfunc
import os
import joblib

from mlflow.tracking import MlflowClient


def export_model():

    mlflow.set_tracking_uri("http://localhost:5000")

    MODEL_NAME = "SolarPowerModel"
    MODEL_STAGE = "Production"

    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

    print(f"Downloading model from: {model_uri}")

    # Load model from MLflow registry
    model = mlflow.pyfunc.load_model(model_uri)

    # Create folder inside app
    os.makedirs("app/model", exist_ok=True)

    # Save as pickle for app usage
    joblib.dump(model, "app/model/model.pkl")

    print("Model exported successfully to app/model/model.pkl")

    return "EXPORTED"


if __name__ == "__main__":
    export_model()