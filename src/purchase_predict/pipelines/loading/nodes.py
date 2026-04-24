"""
This is a boilerplate pipeline 'loading'
generated using Kedro 1.3.1
"""

import os
import tempfile

import pandas as pd
from google.cloud import storage


def load_csv_from_bucket(project: str, bucket_path: str) -> pd.DataFrame:
    """
    Loads a single CSV file from Google Cloud Storage.
    bucket_path format: "bucket-name/path/to/file.csv"
    """
    storage_client = storage.Client()

    # Sépare le nom du bucket et le chemin du fichier
    parts = bucket_path.split("/", 1)
    bucket_name = parts[0]
    file_path = parts[1]

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)

    # Télécharge dans un fichier temporaire
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name

    blob.download_to_filename(tmp_path)
    df = pd.read_csv(tmp_path)
    os.unlink(tmp_path)

    return df
