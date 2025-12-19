from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from api.model import load_model

app = FastAPI(title="Bike Forecast API")

model = load_model()

# -------------------------
# Schéma d'entrée
# -------------------------
class PredictionRequest(BaseModel):
    current_bikes: int
    stationcode: int
    hour: int
    dayofweek: int


# -------------------------
# Endpoint test
# -------------------------
@app.get("/")
def healthcheck():
    return {"status": "ok"}


# -------------------------
# Endpoint prédiction
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
