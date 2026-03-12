"""Gold layer assets - ML datasets."""

import json
from pathlib import Path

from dagster import AssetExecutionContext, Output, TableColumn, TableSchema, asset

from src.assets.gold.datasets import build_datasets
from src.assets.gold.schema import GOLD_SCHEMA
from src.ducklake import get_client


@asset(
    group_name="gold",
    description="Création des datasets train/test ML avec encodage",
    compute_kind="sklearn",
    deps=["full_dataset"],
    required_resource_keys={"mlflow"},
)
def ml_datasets(context: AssetExecutionContext) -> Output[dict]:
    """
    Crée gold.train et gold.test avec encodage et split stratifié.

    Pipeline:
    1. Charge silver.full_dataset
    2. Encode 'atm' avec LabelEncoder
    3. Drop NA
    4. Split 80/20 stratifié sur target
    5. Sauvegarde encoders sur S3 (atm_encoder.pkl, features.pkl)
    6. Crée gold.train, gold.test, gold.feature_metadata

    Tables: gold.train, gold.test, gold.feature_metadata
    Artifacts S3: atm_encoder.pkl, features.pkl
    """
    context.log.info("🔄 Construction datasets ML...")
    result = build_datasets()

    train_pos_pct = (result["train_positives"] / result["train_rows"]) * 100
    test_pos_pct = (result["test_positives"] / result["test_rows"]) * 100

    context.log.info(
        f"✅ Datasets créés: "
        f"train={result['train_rows']} ({result['train_positives']} positifs [{train_pos_pct:.1f}%]), "
        f"test={result['test_rows']} ({result['test_positives']} positifs [{test_pos_pct:.1f}%])"
    )

    # Log artefacts de preprocessing dans MLflow pour consommation runtime (Init Container)
    mlflow = context.resources.mlflow
    for artifact_name in ["atm_encoder.pkl", "features.pkl", "kmeans_geo.pkl"]:
        artifact_path = Path(artifact_name)
        if artifact_path.exists():
            mlflow.log_artifact(str(artifact_path))
            context.log.info(f"📦 Artefact MLflow loggé: {artifact_name}")

    # Métadonnées table/colonnes Dagster (alignées silver)
    client = get_client()
    conn = client.conn
    sample_df = conn.execute(f"SELECT * FROM {GOLD_SCHEMA}.train LIMIT 10").df()

    sample_json = json.dumps(sample_df.to_dict(orient="records"), default=str)
    columns = [
        TableColumn(name=str(col), type=str(dtype)) for col, dtype in sample_df.dtypes.items()
    ]
    table_schema = TableSchema(columns=columns)

    return Output(
        result,
        metadata={
            "train_rows": result["train_rows"],
            "test_rows": result["test_rows"],
            "train_positives": result["train_positives"],
            "test_positives": result["test_positives"],
            "train_positive_rate": f"{train_pos_pct:.2f}%",
            "test_positive_rate": f"{test_pos_pct:.2f}%",
            "features_count": result["total_features"],
            "split_ratio": "80/20",
            "tables": "gold.train, gold.test, gold.feature_metadata",
            "s3_artifacts": "atm_encoder.pkl, features.pkl, kmeans_geo.pkl",
            "n_geo_clusters": result["n_geo_clusters"],
            "geo_cluster_coverage": result["geo_cluster_coverage"],
            "sample_10_rows": sample_json,
            "dagster/row_count": result["train_rows"],
            "dagster/table_name": f"{GOLD_SCHEMA}.train",
            "dagster/column_schema": table_schema,
        },
    )
