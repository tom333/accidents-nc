"""Dagster definitions for accidents pipeline."""

from dagster import Definitions, load_assets_from_modules
from dagster_pipeline import assets

# Load all assets
all_assets = load_assets_from_modules([assets])

# Define Dagster definitions
defs = Definitions(
    assets=all_assets,
)
