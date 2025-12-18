from kedro.pipeline import Pipeline, node
from .nodes import build_supervised_dataset

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=build_supervised_dataset,
                inputs=dict(
                    cleaned_data="cleaned_data",
                    horizons="params:horizons"
                ),
                outputs="supervised_data",
                name="build_supervised_dataset",
            )
        ]
    )
