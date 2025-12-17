import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_model(cleaned_data, params):
    X = cleaned_data.drop("target", axis=1)
    y = cleaned_data["target"]

    model = LogisticRegression(**params)

    with mlflow.start_run():
        model.fit(X, y)
        preds = model.predict(X)

        acc = accuracy_score(y, preds)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_params(params)
        mlflow.sklearn.log_model(model, "model")

    return model
