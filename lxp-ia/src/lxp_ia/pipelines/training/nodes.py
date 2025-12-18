import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def train_regression_model(df: pd.DataFrame):
    # ------------------------------------------------------------------
    # 1. Préparation des features
    # ------------------------------------------------------------------
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek

    FEATURES = [
        "current_bikes",
        "stationcode",
        "hour",
        "dayofweek",
    ]

    TARGETS = [
        "target_h1",
        "target_h3",
        "target_h6",
    ]

    X = df[FEATURES]
    y = df[TARGETS]

    # ------------------------------------------------------------------
    # 2. Split train / test
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # ------------------------------------------------------------------
    # 3. Modèle multi-output
    # ------------------------------------------------------------------
    base_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
    )

    model = MultiOutputRegressor(base_model)

    # ------------------------------------------------------------------
    # 4. Entraînement + MLflow
    # ------------------------------------------------------------------
    with mlflow.start_run(run_name="rf_multi_horizon_v1"):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = {}
        for i, horizon in enumerate(TARGETS):
            mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
            rmse = np.sqrt(
                mean_squared_error(
                    y_test.iloc[:, i],
                    y_pred[:, i],
                )
            )


            metrics[f"MAE_{horizon}"] = mae
            metrics[f"RMSE_{horizon}"] = rmse

        mlflow.log_params({
            "model_type": "RandomForest",
            "n_estimators": 200,
            "max_depth": 15,
        })

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

    return model
