"""Assets Dagster organisés selon l'architecture médaillons (bronze/silver/gold).

- Bronze : données brutes ingérées depuis data.gouv.fr
- Silver : features enrichies avec OSM, datetime, météo
- Gold : datasets train/test et modèles ML entraînés
"""
from dagster import AssetExecutionContext, asset

from pipeline.stage_ingest import ingest_all
from pipeline.stage_features import build_feature_store
from pipeline.stage_datasets import build_datasets
from pipeline.stage_modeling import run_training


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
# GOLD LAYER - Datasets ML et modèles
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


@asset(
	name="gold_models",
	key_prefix=["gold"],
	description="[GOLD] Modèles ML entraînés (RF, LGBM, XGB, CatBoost, TabNet, MLP) + MLflow tracking",
	deps=[gold_datasets],
	group_name="gold",
)
def gold_models(context: AssetExecutionContext) -> dict:
	"""Couche gold : entraînement multi-modèles avec RandomizedSearchCV, sélection du meilleur (AUC-ROC), sauvegarde accident_model.pkl."""
	summary = run_training()
	context.log.info(f"[GOLD] Modélisation terminée: best_model={summary.get('best_model')}")
	return summary
