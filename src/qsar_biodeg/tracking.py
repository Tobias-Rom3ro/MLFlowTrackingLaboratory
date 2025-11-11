import mlflow
import mlflow.tensorflow
from contextlib import contextmanager
from typing import Dict, Optional


def set_experiment(name: str):
    mlflow.set_experiment(name)


@contextmanager
def run(run_name: str, tags: Optional[Dict[str, str]] = None, nested: bool = False):
    with mlflow.start_run(run_name=run_name, nested=nested):
        if tags:
            mlflow.set_tags(tags)
        yield


def log_params(params: Dict[str, any]):
    for key, value in params.items():
        mlflow.log_param(key, value)


def log_metrics(metrics: Dict[str, float]):
    for key, value in metrics.items():
        mlflow.log_metric(key, value)


def log_metric(key: str, value: float):
    mlflow.log_metric(key, value)


def log_artifact(path: str):
    mlflow.log_artifact(path)


def enable_tf_autolog():
    mlflow.tensorflow.autolog()


def enable_global_autolog():
    mlflow.autolog()