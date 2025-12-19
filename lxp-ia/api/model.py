import mlflow
import mlflow.sklearn

MODEL_NAME = "bike_forecast_multi_horizon"

def load_model():
    model_uri = f"models:/{MODEL_NAME}/Production"
    model = mlflow.sklearn.load_model(model_uri)
    return model
