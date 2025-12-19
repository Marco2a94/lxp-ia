from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from api.model import load_model
from api.agent import ask_agent

app = FastAPI(title="Bike Forecast API")

model = load_model()

# -------------------------
# Schémas
# -------------------------
class PredictionRequest(BaseModel):
    current_bikes: int
    stationcode: int
    hour: int
    dayofweek: int


class AskRequest(BaseModel):
    question: str

# -------------------------
# Healthcheck
# -------------------------
@app.get("/")
def healthcheck():
    return {"status": "ok"}


# -------------------------
# Prédiction brute ML
# -------------------------
@app.post("/predict")
def predict(req: PredictionRequest):
    X = pd.DataFrame([req.dict()])
    preds = model.predict(X)[0]

    return {
        "horizon_1h": round(preds[0], 2),
        "horizon_3h": round(preds[1], 2),
        "horizon_6h": round(preds[2], 2),
    }


# -------------------------
# Agent IA (langage naturel)
# -------------------------
@app.post("/ask")
def ask(req: AskRequest):
    answer = ask_agent(req.question)
    return {
        "answer": answer
    }

