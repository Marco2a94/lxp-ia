from kedro.pipeline import Pipeline, node
from .nodes import train_model

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=train_model,
            inputs=["cleaned_data", "params:model"],
            outputs="model",
            name="train_model_node",
        )
    ])
