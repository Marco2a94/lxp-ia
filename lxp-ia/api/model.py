import mlflow
import mlflow.sklearn

MODEL_NAME = "bike_forecast_multi_horizon"

def load_model():
    try:
        model_uri = f"models:/{MODEL_NAME}/Production"
        return mlflow.sklearn.load_model(model_uri)
    except Exception:
        # fallback : dernière version disponible
        model_uri = f"models:/{MODEL_NAME}/latest"
        return mlflow.sklearn.load_model(model_uri)
