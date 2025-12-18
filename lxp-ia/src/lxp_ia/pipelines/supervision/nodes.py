import pandas as pd

def build_supervised_dataset(df, horizons):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # TRI ABSOLU REQUIS POUR merge_asof
    df = df.sort_values(["stationcode", "timestamp"]).reset_index(drop=True)

    supervised = []

    for horizon in horizons:
        df_h = df.copy()

        df_h["horizon"] = horizon
        df_h["target_timestamp"] = df_h["timestamp"] + pd.Timedelta(minutes=horizon)

        # TRI ABSOLU REQUIS POUR merge_asof
        df_h = df_h.sort_values(
            ["stationcode", "target_timestamp"]
        ).reset_index(drop=True)

        merged = pd.merge_asof(
            df_h,
            df[["stationcode", "timestamp", "numbikesavailable"]],
            left_on="target_timestamp",
            right_on="timestamp",
            by="stationcode",
            direction="nearest"
        )

        merged = merged.rename(columns={
            "timestamp_x": "timestamp",
            "numbikesavailable": "target"
        })

        merged = merged.drop(columns=["timestamp_y", "target_timestamp"])

        supervised.append(merged)

    return pd.concat(supervised, ignore_index=True)
