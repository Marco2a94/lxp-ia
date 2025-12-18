import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def train_regression_model(df):
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek

    features = [
        "hour",
        "dayofweek",
        "horizon",
        "current_bikes"
    ]

    X = df[features]
    y = df["target"]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    return model
