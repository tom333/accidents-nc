"""Assets Dagster pour la production (bronze/silver/gold datasets uniquement).

Cette version exclut gold_models qui nécessite les dépendances training (optuna, catboost, etc.).
"""
from dagster import AssetExecutionContext, asset

from pipeline.stage_ingest import ingest_all
from pipeline.stage_features import build_feature_store
from pipeline.stage_datasets import build_datasets


# ============================================================================
# BRONZE LAYER - Données brutes
# ============================================================================

@asset(
    name="bronze_accidents_nc",
    key_prefix=["bronze"],
    description="[BRONZE] Ingestion des CSV data.gouv.fr (caractéristiques + usagers) → bronze.accidents_nc",
    group_name="bronze",
)
def bronze_accidents_nc(context: AssetExecutionContext) -> dict:
    """Couche bronze : données brutes accident New Caledonia (dep=988) 2019-2024."""
    stats = ingest_all()
    context.log.info(f"[BRONZE] Ingestion terminée: {stats}")
    return stats


# ============================================================================
# SILVER LAYER - Données enrichies et nettoyées
# ============================================================================

@asset(
    name="silver_features",
    key_prefix=["silver"],
    description="[SILVER] Features enrichies (positifs + négatifs synthétiques, OSM, datetime) → silver.features",
    deps=[bronze_accidents_nc],
    group_name="silver",
)
def silver_features(context: AssetExecutionContext) -> dict:
    """Couche silver : features temporelles, routières (buffer 200m), négatifs synthétiques (ratio 22k)."""
    stats = build_feature_store()
    context.log.info(f"[SILVER] Features construites: {stats}")
    return stats


# ============================================================================
# GOLD LAYER - Datasets ML uniquement (pas de training)
# ============================================================================

@asset(
    name="gold_datasets",
    key_prefix=["gold"],
    description="[GOLD] Datasets train/test (80/20 split) + metadata → gold.train / gold.test",
    deps=[silver_features],
    group_name="gold",
)
def gold_datasets(context: AssetExecutionContext) -> dict:
    """Couche gold : datasets train/test avec encodage atm, sauvegarde des artefacts (atm_encoder.pkl, features.pkl)."""
    stats = build_datasets()
    context.log.info(f"[GOLD] Datasets préparés: {stats}")
    return stats
