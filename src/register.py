import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("file:///C:/Users/Ritesh gawade/Desktop/MSC AI/proj/Solar_Power_Generation_Forecasting/mlruns")

client = MlflowClient()

experiment = client.get_experiment_by_name("SolarPowerForecasting")

# ✅ Get latest runs only (sorted)
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=20
)

best_run = None
best_score = -float("inf")

for run in runs:
    val_r2 = run.data.metrics.get("val_r2")

    if val_r2 is not None and val_r2 > best_score:
        best_score = val_r2
        best_run = run

if best_run is None:
    raise Exception("No valid runs found!")

best_run_id = best_run.info.run_id
model_uri = f"runs:/{best_run_id}/model"

print(f"Best Run ID: {best_run_id}")
print(f"Best Validation R2: {best_score}")

model_name = "SolarPowerModel"

registered_model = mlflow.register_model(model_uri, model_name)

# Move to Production
client.transition_model_version_stage(
    name=model_name,
    version=registered_model.version,
    stage="Production"
)

print("********Model registered and moved to Production********")