import pandas as pd

def build_supervised_dataset(df, horizons):
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # OBLIGATOIRE
    df = df.sort_values(["stationcode", "timestamp"])

    supervised = []

    for horizon in horizons:
        df_h = df.copy()
        df_h["horizon"] = horizon
        df_h["target_timestamp"] = df_h["timestamp"] + pd.Timedelta(minutes=horizon)

        # OBLIGATOIRE
        df_h = df_h.sort_values(["stationcode", "target_timestamp"])

        merged = pd.merge_asof(
            df_h,
            df[["stationcode", "timestamp", "numbikesavailable"]],
            left_on="target_timestamp",
            right_on="timestamp",
            by="stationcode",
            direction="nearest"
        )

        merged = merged.rename(columns={
            "numbikesavailable_y": "target"
        })

        supervised.append(merged)

    return pd.concat(supervised, ignore_index=True)
