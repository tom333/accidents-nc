
"""Assets package - load all asset modules."""

from dagster import load_assets_from_modules
from dagster_pipeline.assets import (
	blend_deepchecks, bronze, silver, gold,
	catboost, xgboost, mlp,
	blending, blend_eval, blend_export
)

# Export all assets
bronze_assets = load_assets_from_modules([bronze])
silver_assets = load_assets_from_modules([silver])
gold_assets = load_assets_from_modules([
	gold, catboost, xgboost, mlp, blending, blend_eval, blend_export, blend_deepchecks
])
