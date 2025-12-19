import requests
from openai import OpenAI

PREDICT_API_URL = "http://127.0.0.1:8000/predict"

client = OpenAI()  # nécessite OPENAI_API_KEY

def ask_agent(
    question: str,
    current_bikes: int,
    stationcode: int,
    hour: int,
    dayofweek: int,
):
    payload = {
        "current_bikes": current_bikes,
        "stationcode": stationcode,
        "hour": hour,
        "dayofweek": dayofweek,
    }

    response = requests.post(PREDICT_API_URL, json=payload)
    preds = response.json()

    h3 = preds["horizon_3h"]

    if h3 <= 2:
        risk = "élevé"
    elif h3 <= 5:
        risk = "modéré"
    else:
        risk = "faible"

    return (
        f"Dans 3 heures, la station devrait avoir environ {h3:.1f} vélos. "
        f"Le risque qu'elle soit vide est donc {risk}."
    )
