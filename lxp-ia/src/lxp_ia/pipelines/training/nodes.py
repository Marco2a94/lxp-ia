import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


def train_regression_model(df):
    df = df.copy()

    # Features temporelles
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek

    features = [
        "stationcode",
        "hour",
        "dayofweek",
        "horizon",
        "numbikesavailable"
    ]

    X = df[features]
    y = df["target"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("station", OneHotEncoder(handle_unknown="ignore"), ["stationcode"]),
        ],
        remainder="passthrough",
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    tscv = TimeSeriesSplit(n_splits=3)
    maes = []

    with mlflow.start_run():
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

            mae = mean_absolute_error(y_test, preds)
            maes.append(mae)

        mlflow.log_metric("mae_mean", sum(maes) / len(maes))
        mlflow.sklearn.log_model(pipeline, "model")

    return pipeline
