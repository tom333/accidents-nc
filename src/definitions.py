"""Dagster definitions for accidents pipeline."""

import os

from dagster import Definitions, load_assets_from_package_module
from dagster_mlflow import mlflow_tracking
from dotenv import load_dotenv

from src import assets

# Load environment variables from .env file
load_dotenv()

# Load all assets recursively from the assets package
all_assets = load_assets_from_package_module(assets)

# MLflow resource configuration
mlflow_resource = mlflow_tracking.configured(
    {
        "experiment_name": "accidents-nc",
        "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        "env": {
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", ""),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            "MLFLOW_S3_ENDPOINT_URL": os.getenv(
                "MLFLOW_S3_ENDPOINT_URL", os.getenv("AWS_ENDPOINT_URL", "https://rustfs.tgu.ovh")
            ),
        },
    }
)

# Define Dagster definitions
defs = Definitions(assets=all_assets, resources={"mlflow": mlflow_resource})
