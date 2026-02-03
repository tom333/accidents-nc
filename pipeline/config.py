from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import os

import duckdb


DUCKLAKE_DB_ALIAS = os.getenv("DUCKLAKE_DB_ALIAS", "accidents_ducklake")

# Architecture médaillons (Medallion Architecture)
BRONZE_SCHEMA = "bronze"  # Données brutes ingérées (accidents_nc)
SILVER_SCHEMA = "silver"  # Données enrichies et nettoyées (features)
GOLD_SCHEMA = "gold"      # Données agrégées et datasets ML (train/test/models)


@dataclass(frozen=True)
class PipelineParams:
    n_negative_samples_ratio: int = 22000
    buffer_meters: int = 200
    grid_step: float = 0.02
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    search_iterations: int = 20
    accident_exclusion_buffer_km: float = 0.3
    temporal_risk_ratio: float = 0.85


PIPELINE_PARAMS = PipelineParams()


def ensure_connection() -> duckdb.DuckDBPyConnection:
    """Retourne une connexion DuckDB attachée au catalogue DuckLake.

    La configuration du catalogue et du stockage est pilotée par les variables
    d'environnement suivantes :

    - DUCKLAKE_DATABASE_URL : DSN DuckLake côté catalogue, par exemple
      "postgres:dbname=ducklake_catalog host=postgresql.datalab.svc.cluster.local".
    - DUCKLAKE_DATA_PATH : chemin de stockage des fichiers Parquet, par exemple
      un préfixe S3 géré par RustFS (ex: "s3://bucket-name/ducklake/").
    - AWS_ENDPOINT_URL : endpoint S3 compatible (ex: "http://rustfs-svc.ia-lab.svc.cluster.local:9000").
    - AWS_ACCESS_KEY_ID : clé d'accès S3.
    - AWS_SECRET_ACCESS_KEY : secret S3.
    - AWS_REGION : région S3 (optionnel, par défaut "us-east-1").
    """

    if hasattr(ensure_connection, "_conn") and getattr(ensure_connection, "_conn") is not None:
        return getattr(ensure_connection, "_conn")

    conn = duckdb.connect()

    conn.execute("INSTALL ducklake;")
    conn.execute("LOAD ducklake;")

    ducklake_catalog = os.getenv("DUCKLAKE_DATABASE_URL")
    if not ducklake_catalog:
        raise RuntimeError(
            "DUCKLAKE_DATABASE_URL n'est pas défini. "
            "Exemple pour PostgreSQL : "
            "postgres:dbname=ducklake_catalog host=postgresql.datalab.svc.cluster.local"
        )

    if ducklake_catalog.startswith("postgres:"):
        conn.execute("INSTALL postgres;")
        conn.execute("LOAD postgres;")

    data_path = os.getenv("DUCKLAKE_DATA_PATH", "ducklake_files/")
    
    # Créer un secret S3 si le data_path commence par s3://
    # Conforme aux recommandations officielles DuckLake : créer le secret AVANT l'ATTACH
    if data_path.startswith("s3://"):
        endpoint = os.getenv("AWS_ENDPOINT_URL", "")
        key_id = os.getenv("AWS_ACCESS_KEY_ID", "")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
        region = os.getenv("AWS_REGION", "us-east-1")
        
        if not (endpoint and key_id and secret_key):
            raise RuntimeError(
                "Pour utiliser un DATA_PATH S3, définir AWS_ENDPOINT_URL, "
                "AWS_ACCESS_KEY_ID et AWS_SECRET_ACCESS_KEY"
            )
        
        # Déterminer use_ssl et url_style depuis l'endpoint
        use_ssl = endpoint.startswith("https://")
        endpoint_clean = endpoint.replace("https://", "").replace("http://", "")
        
        conn.execute(
            f"""
            CREATE SECRET ducklake_s3_secret (
                TYPE S3,
                KEY_ID '{key_id}',
                SECRET '{secret_key}',
                ENDPOINT '{endpoint_clean}',
                REGION '{region}',
                USE_SSL {use_ssl},
                URL_STYLE 'path'
            );
            """
        )

    attach_target = f"ducklake:{ducklake_catalog}"

    conn.execute(
        f"ATTACH '{attach_target}' AS {DUCKLAKE_DB_ALIAS} (DATA_PATH '{data_path}');"
    )
    conn.execute(f"USE {DUCKLAKE_DB_ALIAS};")

    for schema in (BRONZE_SCHEMA, SILVER_SCHEMA, GOLD_SCHEMA):
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

    setattr(ensure_connection, "_conn", conn)
    return conn
