"""Dagster definitions avec assets organisés en architecture médaillons (bronze/silver/gold)."""
from dagster import Definitions

from . import assets_prod as _assets


defs = Definitions(
    assets=[
        _assets.bronze_accidents_nc,
        _assets.silver_features,
        _assets.gold_datasets,
    ],
)

