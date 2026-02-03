"""Package pipeline avec architecture médaillons (bronze/silver/gold)."""
from .config import (
    PIPELINE_PARAMS,
    BRONZE_SCHEMA,
    DUCKLAKE_DB_ALIAS,
    SILVER_SCHEMA,
    GOLD_SCHEMA,
    ensure_connection,
)

__all__ = [
    "PIPELINE_PARAMS",
    "BRONZE_SCHEMA",
    "DUCKLAKE_DB_ALIAS",
    "SILVER_SCHEMA",
    "GOLD_SCHEMA",
    "ensure_connection",
]
