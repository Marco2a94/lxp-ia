# src/lxp_ia/pipelines/training/nodes.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_regression_model(
    supervised_data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Train a multi-output regression model for multi-horizon bike availability prediction.
    """

    df = supervised_data.copy()

    # ------------------------------------------------------------------
    # 1. Sécurité & typage
    # ------------------------------------------------------------------
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # ------------------------------------------------------------------
    # 2. Features temporelles
    # ------------------------------------------------------------------
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek

    feature_cols = [
        "hour",
        "dayofweek",
        "current_bikes",
    ]

    # ------------------------------------------------------------------
    # 3. Construction des targets multi-horizon
    # ------------------------------------------------------------------
    target_df = (
        df.pivot_table(
            index=["stationcode", "timestamp"],
            columns="horizon",
            values="target",
        )
        .reset_index()
        .sort_values("timestamp")
    )

    # On garde uniquement les lignes complètes
    horizon_cols = [c for c in target_df.columns if isinstance(c, int)]
    target_df = target_df.dropna(subset=horizon_cols)

    df = df.merge(
        target_df,
        on=["stationcode", "timestamp"],
        how="inner",
    )

    X = df[feature_cols]
    y = df[horizon_cols]

    # ------------------------------------------------------------------
    # 4. Split temporel (PAS aléatoire)
    # ------------------------------------------------------------------
    split_index = int(len(df) * (1 - test_size))

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # ------------------------------------------------------------------
    # 5. Modèle
    # ------------------------------------------------------------------
    base_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=random_state,
        n_jobs=-1,
    )

    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 6. Métriques de base (provisoires)
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test)

    metrics = {}
    for i, h in enumerate(horizon_cols):
        metrics[f"mae_h{h}"] = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
        metrics[f"rmse_h{h}"] = mean_squared_error(
            y_test.iloc[:, i],
            y_pred[:, i],
            squared=False,
        )

    return {
        "model": model,
        "metrics": metrics,
        "horizons": horizon_cols,
        "features": feature_cols,
    }
