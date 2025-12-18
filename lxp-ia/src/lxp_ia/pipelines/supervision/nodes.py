import pandas as pd

def build_supervised_dataset(cleaned_data: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    df = cleaned_data.copy()

    # Sécurité minimale
    if "timestamp" not in df.columns:
        raise ValueError("cleaned_data must contain a 'timestamp' column")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["stationcode", "timestamp"])

    supervised_rows = []

    for horizon in horizons:
        df_h = df.copy()
        df_h["target_timestamp"] = df_h["timestamp"] + pd.Timedelta(minutes=horizon)

        # Auto-jointure temporelle par station
        merged = pd.merge_asof(
            df_h,
            df[["stationcode", "timestamp", "numbikesavailable"]],
            left_on="target_timestamp",
            right_on="timestamp",
            by="stationcode",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=5),
            suffixes=("", "_future")
        )

        merged = merged.dropna(subset=["numbikesavailable_future"])
        merged["horizon_minutes"] = horizon
        merged["target"] = merged["numbikesavailable_future"]

        supervised_rows.append(merged)

    supervised_df = pd.concat(supervised_rows, ignore_index=True)

    # Nettoyage final
    supervised_df = supervised_df.drop(columns=[
        "target_timestamp",
        "numbikesavailable_future"
    ])

    return supervised_df
