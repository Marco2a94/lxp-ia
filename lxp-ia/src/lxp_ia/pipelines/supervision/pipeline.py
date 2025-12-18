from kedro.pipeline import Pipeline, node
from .nodes import build_supervised_dataset


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=build_supervised_dataset,
                inputs={
                    "cleaned_data": "cleaned_data",
                    "horizons": "params:horizons",
                },
                outputs={
                    "model": "regression_model",
                    "metrics": "training_metrics",
                },
                name="build_supervised_dataset",
            )
        ]
    )
