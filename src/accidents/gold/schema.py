"""Schémas gold (datasets ML et modèles)."""

# Nom du schéma gold dans DuckLake
GOLD_SCHEMA = "ducklake.gold"

# Tables gold:
# - gold.train : Dataset entraînement (80%)
# - gold.test : Dataset test (20%)
# - gold.feature_metadata : Métadonnées des features
# - gold_models : Modèles entraînés (si training activé)

# Artefacts S3:
# - s3://accidents-bucket/cache/atm_encoder.pkl
# - s3://accidents-bucket/cache/features.pkl
# - s3://accidents-bucket/cache/accident_model.pkl (si training)
