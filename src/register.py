import mlflow
from mlflow.tracking import MlflowClient

def run_registry():

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = MlflowClient()

    experiment = client.get_experiment_by_name("SolarPowerForecasting")

    if experiment is None:
        print("Experiment not found!")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.val_r2 DESC"],
        max_results=1
    )

    best_run = runs[0]
    best_run_id = best_run.info.run_id

    # ================= FULL MODEL REPORT =================
    print("\n================ BEST MODEL SELECTED ================")
    print(f"Model Name   : SolarPowerModel")
    print(f"Run ID       : {best_run_id}")
    print(f"Experiment   : {experiment.name}")
    print(f"Tracking URI : http://127.0.0.1:5000")

    print("\n---------------- PARAMETERS ----------------")
    if best_run.data.params:
        for k, v in sorted(best_run.data.params.items()):
            print(f"{k} : {v}")
    else:
        print("No parameters logged.")

    print("\n---------------- METRICS ----------------")
    if best_run.data.metrics:
        for k, v in sorted(best_run.data.metrics.items()):
            print(f"{k} : {v}")
    else:
        print("No metrics logged.")

    print("\n---------------- TAGS ----------------")
    if best_run.data.tags:
        for k, v in sorted(best_run.data.tags.items()):
            print(f"{k} : {v}")

    print("=====================================================\n")

    model_uri = f"runs:/{best_run_id}/model"

    print(f"[INFO] Registering model from URI: {model_uri}")

    model_name = "SolarPowerModel"

    registered_model = mlflow.register_model(model_uri, model_name)

    print("\n================ MODEL REGISTERED ================")
    print(f"Model Name   : {model_name}")
    print(f"Version      : {registered_model.version}")
    print(f"Run ID       : {best_run_id}")
    print("=================================================\n")

    client.transition_model_version_stage(
        name=model_name,
        version=registered_model.version,
        stage="Production"
    )

    print(f"[INFO] Model moved to Production stage ✔")

    return "REGISTERED"


if __name__ == "__main__":
    run_registry()
'''
import mlflow
from mlflow.tracking import MlflowClient

# MLflow server (local machine)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

experiment = client.get_experiment_by_name("SolarPowerForecasting")

if experiment is None:
    raise Exception("Experiment not found!")

# Get best run
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.val_r2 DESC"],
    max_results=1
)

if len(runs) == 0:
    raise Exception("No runs found!")

best_run = runs[0]
best_run_id = best_run.info.run_id

print("Best Run ID:", best_run_id)
print("Best Val R2:", best_run.data.metrics.get("val_r2"))

# Register model
model_uri = f"runs:/{best_run_id}/model"
model_name = "SolarPowerModel"

registered_model = mlflow.register_model(model_uri, model_name)

client.transition_model_version_stage(
    name=model_name,
    version=registered_model.version,
    stage="Production"
)

print("MODEL REGISTERED SUCCESSFULLY")
'''