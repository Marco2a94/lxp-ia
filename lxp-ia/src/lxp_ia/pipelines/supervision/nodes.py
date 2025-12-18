import pandas as pd


def build_supervised_dataset(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    supervised_parts = []

    for station, df_station in df.groupby("stationcode"):
        df_station = df_station.sort_values("timestamp")

        for horizon in horizons:
            df_h = df_station.copy()
            df_h["horizon"] = horizon
            df_h["target_timestamp"] = df_h["timestamp"] + pd.Timedelta(minutes=horizon)

            df_h = df_h.sort_values("target_timestamp")

            merged = pd.merge_asof(
                df_h,
                df_station[["timestamp", "numbikesavailable"]],
                left_on="target_timestamp",
                right_on="timestamp",
                direction="nearest",
                allow_exact_matches=True,
            )

            merged = merged.rename(
                columns={"numbikesavailable": "target"}
            )

            merged["stationcode"] = station
            supervised_parts.append(merged)

    supervised = pd.concat(supervised_parts, ignore_index=True)
    return supervised
