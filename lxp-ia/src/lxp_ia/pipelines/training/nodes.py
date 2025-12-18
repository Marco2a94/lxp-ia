import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def train_regression_model(
    supervised_data: pd.DataFrame,
):
    df = supervised_data.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Features temporelles
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek

    feature_cols = [
        "current_bikes",
        "hour",
        "dayofweek",
    ]

    target_cols = [
        c for c in df.columns if c.startswith("target_h")
    ]

    X = df[feature_cols]
    y = df[target_cols]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X, y)

    return model
