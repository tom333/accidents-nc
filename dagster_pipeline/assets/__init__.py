"""Assets package - load all asset modules."""

from dagster import load_assets_from_modules
from dagster_pipeline.assets import bronze, silver, gold

# Export all assets
bronze_assets = load_assets_from_modules([bronze])
silver_assets = load_assets_from_modules([silver])
gold_assets = load_assets_from_modules([gold])
