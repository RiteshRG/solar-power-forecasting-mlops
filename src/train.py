import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def run_training():

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("SolarPowerForecasting")

    version = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.read_csv("data/processed/train.csv")

    X = df.drop("AC_POWER", axis=1)
    y = df["AC_POWER"]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(),
        "GradientBoosting": GradientBoostingRegressor(),
        "XGBoost": XGBRegressor(objective="reg:squarederror")
    }

    for name, model in models.items():

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        with mlflow.start_run(run_name=name):

            mlflow.set_tag("version", version)

            pipeline.fit(X_train, y_train)

            val_pred = pipeline.predict(X_val)
            test_pred = pipeline.predict(X_test)

            mlflow.log_metric("val_r2", r2_score(y_val, val_pred))
            mlflow.log_metric("test_r2", r2_score(y_test, test_pred))
            mlflow.log_metric("rmse", np.sqrt(mean_squared_error(y_test, test_pred)))
            mlflow.log_metric("mae", mean_absolute_error(y_test, test_pred))

            mlflow.log_param("model", name)

            mlflow.sklearn.log_model(pipeline, "model")

    return "TRAINING_DONE"


if __name__ == "__main__":
    run_training()

'''import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("SolarPowerForecasting")

# 👉 Create version (timestamp based)
version = datetime.now().strftime("%Y%m%d_%H%M%S")

df = pd.read_csv("data/processed/train.csv")

X = df.drop("AC_POWER", axis=1)
y = df["AC_POWER"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(),
    "GradientBoosting": GradientBoostingRegressor(),
    "XGBoost": XGBRegressor(objective="reg:squarederror")
}

for name, model in models.items():

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    with mlflow.start_run(run_name=name):

        # ✅ IMPORTANT: tag version
        mlflow.set_tag("version", version)

        pipeline.fit(X_train, y_train)

        train_pred = pipeline.predict(X_train)
        val_pred = pipeline.predict(X_val)
        test_pred = pipeline.predict(X_test)

        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        test_r2 = r2_score(y_test, test_pred)

        rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        mae = mean_absolute_error(y_test, test_pred)

        mlflow.log_param("model", name)

        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("val_r2", val_r2)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(pipeline, "model")

print(f"Training complete for version: {version}")'''