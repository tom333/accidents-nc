"""Job Dagster pour orchestrer le pipeline blending gold."""
from dagster import Definitions, AssetSelection, define_asset_job
from assets.catboost import tune_catboost
from assets.xgboost import tune_xgboost
from assets.mlp import tune_mlp

blend_job = define_asset_job(
    name="gold_blend_job",
    selection=AssetSelection.groups("gold")
)

defs = Definitions(
    assets=[tune_catboost, tune_xgboost, tune_mlp],
    jobs=[blend_job]
)
