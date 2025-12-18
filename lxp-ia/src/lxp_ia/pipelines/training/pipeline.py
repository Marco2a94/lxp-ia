from kedro.pipeline import Pipeline, node
from .nodes import train_regression_model


def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=train_regression_model,
            inputs="supervised_data",
            outputs="regression_model",
            name="train_regression_model",
        )
    ])
