"""
This is a boilerplate pipeline 'deployment'
generated using Kedro 1.3.1
"""

import os
import mlflow
from mlflow.tracking import MlflowClient


def push_to_model_registry(registry_name: str, model_uri: str) -> str:
    """Pushes a model version to the MLflow model registry."""
    tracking_uri = os.getenv("MLFLOW_SERVER")
    if not tracking_uri:
        raise ValueError("MLFLOW_SERVER environment variable is not set")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Crée le modèle dans le registry s'il n'existe pas
    try:
        client.get_registered_model(registry_name)
    except Exception:
        client.create_registered_model(registry_name)

    result = client.create_model_version(
        name=registry_name,
        source=model_uri,
    )
    return result.version


def stage_model(registry_name: str, version: str) -> None:
    """Assigns an alias (staging or production) to a model version."""
    env = os.getenv("ENV")
    if not env:
        return
    client = MlflowClient()
    client.set_registered_model_alias(
        name=registry_name,
        alias=env,
        version=version,
    )
