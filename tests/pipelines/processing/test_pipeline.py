"""
This is a boilerplate test file for pipeline 'loading'
generated using Kedro 1.3.1.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""

from kedro.runner import SequentialRunner
from purchase_predict.pipelines.processing.pipeline import create_pipeline


def test_pipeline(catalog_test):
    runner = SequentialRunner()
    pipeline = create_pipeline()
    output = runner.run(pipeline, catalog_test)
    X_train = output["X_train"].load()
    y_train = output["y_train"].load()
    X_test = output["X_test"].load()
    y_test = output["y_test"].load()
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
