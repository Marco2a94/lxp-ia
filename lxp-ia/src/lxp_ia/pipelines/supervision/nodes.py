import pandas as pd


def build_supervised_dataset(
    cleaned_data: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    """
    Build a multi-output supervised dataset:
    one row per (station, timestamp),
    one target column per horizon.
    """

    df = cleaned_data.copy()

    # Sécurité types
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # On garde uniquement ce qu'il faut
    df = df[
        ["stationcode", "timestamp", "numbikesavailable"]
    ].rename(
        columns={"numbikesavailable": "current_bikes"}
    )

    supervised_parts = []

    for station, g in df.groupby("stationcode"):
        g = g.sort_values("timestamp").reset_index(drop=True)

        base = g[["timestamp", "current_bikes"]].copy()

        for h in horizons:
            base[f"target_h{h}"] = g["current_bikes"].shift(-h)

        base["stationcode"] = station

        supervised_parts.append(base)

    supervised = pd.concat(supervised_parts, ignore_index=True)

    # Nettoyage : on enlève les lignes sans toutes les targets
    target_cols = [f"target_h{h}" for h in horizons]
    supervised = supervised.dropna(subset=target_cols)

    return supervised
