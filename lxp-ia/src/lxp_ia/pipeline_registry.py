from kedro.framework.project import find_pipelines

def register_pipelines():
    pipelines = find_pipelines()
    return pipelines
