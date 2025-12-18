import pandas as pd
from typing import List


def build_supervised_dataset(
    df: pd.DataFrame,
    horizons: List[int],
) -> pd.DataFrame:
    """
    Build a supervised dataset for multi-horizon forecasting.

    Output columns:
    - stationcode
    - timestamp        (current time)
    - horizon
    - current_bikes
    - target           (future bikes)
    """

    df = df.copy()

    required_cols = {"stationcode", "timestamp", "numbikesavailable"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    supervised_parts = []

    for station, df_station in df.groupby("stationcode"):
        df_station = df_station.sort_values("timestamp").reset_index(drop=True)

        for horizon in horizons:
            left = df_station.copy()
            left["horizon"] = horizon
            left["target_timestamp"] = (
                left["timestamp"] + pd.Timedelta(minutes=horizon)
            )

            left = left.sort_values("target_timestamp").reset_index(drop=True)

            right = (
                df_station[["timestamp", "numbikesavailable"]]
                .sort_values("timestamp")
                .reset_index(drop=True)
            )

            merged = pd.merge_asof(
                left,
                right,
                left_on="target_timestamp",
                right_on="timestamp",
                direction="nearest",
                tolerance=pd.Timedelta(minutes=5),
            )

            # 🔧 RENOMMAGES EXPLICITES (LA CLÉ DU BUG)
            merged = merged.rename(
                columns={
                    "timestamp_x": "timestamp",
                    "numbikesavailable_x": "current_bikes",
                    "numbikesavailable_y": "target",
                }
            )

            merged["stationcode"] = station

            supervised_parts.append(
                merged[
                    [
                        "stationcode",
                        "timestamp",
                        "horizon",
                        "current_bikes",
                        "target",
                    ]
                ]
            )

    supervised_df = (
        pd.concat(supervised_parts, ignore_index=True)
        .dropna()
        .reset_index(drop=True)
    )

    return supervised_df
