"""
This is a boilerplate test file for pipeline 'loading'
generated using Kedro 1.3.1.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""

import pandas as pd
from kedro.runner import SequentialRunner

from purchase_predict.pipelines.loading.pipeline import create_pipeline


def test_pipeline(catalog_test):
    runner = SequentialRunner()
    pipeline = create_pipeline()
    output = runner.run(pipeline, catalog_test)
    df = output["primary"].load()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert "purchased" in df.columns
