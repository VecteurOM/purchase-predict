import pytest
from kedro.io import DataCatalog, MemoryDataset


@pytest.fixture(scope="module")
def project_id():
    return "project-7c818ee0-8f39-47d7-85d"


@pytest.fixture(scope="module")
def primary_folder():
    return "example-mlops/primary.csv"  # ton vrai fichier


@pytest.fixture(scope="module")
def catalog_test(project_id, primary_folder):
    catalog = DataCatalog(
        {
            "params:gcp_project_id": MemoryDataset(project_id),
            "params:gcs_primary_folder": MemoryDataset(primary_folder),
        }
    )
    return catalog
