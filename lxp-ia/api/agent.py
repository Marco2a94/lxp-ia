# api/agent.py
import requests
from openai import OpenAI

client = OpenAI()

PREDICT_API_URL = "http://127.0.0.1:8000/predict"

def ask_agent(
    question: str,
    current_bikes: int,
    stationcode: int,
    hour: int,
    dayofweek: int,
):
    # 1. Appel à ton modèle ML
    payload = {
        "current_bikes": current_bikes,
        "stationcode": stationcode,
        "hour": hour,
        "dayofweek": dayofweek,
    }

    response = requests.post(PREDICT_API_URL, json=payload)
    preds = response.json()

    # 2. Prompt LLM
    prompt = f"""
Un utilisateur pose la question suivante :
"{question}"

Voici les prédictions du modèle :
- Dans 1h : {preds["horizon_1h"]:.1f} vélos
- Dans 3h : {preds["horizon_3h"]:.1f} vélos
- Dans 6h : {preds["horizon_6h"]:.1f} vélos

Explique la situation simplement, en français, sans jargon technique.
"""

    # 3. Appel LLM
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    return completion.choices[0].message.content
