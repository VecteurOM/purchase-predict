"""
This is a boilerplate pipeline 'loading'
generated using Kedro 1.3.1
"""

from kedro.pipeline import Node, Pipeline

from .nodes import load_csv_from_bucket


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                load_csv_from_bucket,
                ["params:gcp_project_id", "params:gcs_primary_folder"],
                "primary",
            )
        ]
    )
