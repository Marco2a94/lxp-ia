from lxp_ia.pipelines.training.pipeline import create_pipeline

def register_pipelines():
    return {
        "__default__": create_pipeline()
    }
