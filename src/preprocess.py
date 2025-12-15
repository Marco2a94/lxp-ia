# src/preprocess.py
import pandas as pd

RAW_FILE = "data/velib_raw.csv"
OUTPUT_FILE = "data/velib_features.csv"

def main():
    df = pd.read_csv(RAW_FILE)

    # --- Parsing du temps ---
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["stationcode", "timestamp"])

    # --- Filtrage stations actives ---
    df = df[
        (df["is_installed"] == True) &
        (df["is_renting"] == True) &
        (df["is_returning"] == True)
    ]

    # --- Features temporelles ---
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # --- Lags (par station) ---
    df["bikes_lag_1"] = df.groupby("stationcode")["numbikesavailable"].shift(1)
    df["bikes_lag_2"] = df.groupby("stationcode")["numbikesavailable"].shift(2)
    df["bikes_lag_3"] = df.groupby("stationcode")["numbikesavailable"].shift(3)

    # --- Target : T+30 min (si collecte toutes les 5 min → shift -6) ---
    df["target_bikes_t30"] = df.groupby("stationcode")["numbikesavailable"].shift(-6)

    # --- Suppression des lignes incomplètes ---
    df = df.dropna()

    # --- Sélection finale ---
    features = [
        "stationcode",
        "hour",
        "dayofweek",
        "is_weekend",
        "bikes_lag_1",
        "bikes_lag_2",
        "bikes_lag_3",
        "target_bikes_t30",
    ]

    df = df[features]

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Préprocessing terminé : {df.shape[0]} lignes sauvegardées")

if __name__ == "__main__":
    main()
