"""Bronze layer assets - Raw data ingestion."""

import json

from dagster import AssetExecutionContext, Output, TableColumn, TableSchema, asset

from src.assets.bronze.ingest import (
    create_accidents_nc,
    ingest_caracteristiques,
    ingest_usagers,
)
from src.assets.bronze.schema import BRONZE_SCHEMA
from src.ducklake import get_client


def _build_table_metadata(table_name: str) -> tuple[str, TableSchema]:
    client = get_client()
    conn = client.conn
    sample_df = conn.execute(f"SELECT * FROM {BRONZE_SCHEMA}.{table_name} LIMIT 10").df()

    sample_json = json.dumps(sample_df.to_dict(orient="records"), default=str)
    columns = [
        TableColumn(name=str(col), type=str(dtype)) for col, dtype in sample_df.dtypes.items()
    ]
    table_schema = TableSchema(columns=columns)

    return sample_json, table_schema


@asset(
    group_name="bronze",
    description="Ingestion des caractéristiques d'accidents depuis data.gouv.fr (2019-2024)",
    compute_kind="duckdb",
)
def caracteristiques(context: AssetExecutionContext) -> Output[dict]:
    """
    Ingère les données caractéristiques dans bronze.caracteristiques.

    Source: data.gouv.fr CSV union 2019-2024
    Table: bronze.caracteristiques
    """
    context.log.info("🔄 Ingestion caractéristiques...")
    result = ingest_caracteristiques()
    context.log.info(f"✅ Ingéré {result['rows']} lignes caractéristiques")

    sample_json, table_schema = _build_table_metadata("caracteristiques")

    return Output(
        result,
        metadata={
            "rows": result["rows"],
            "table": "bronze.caracteristiques",
            "source": "data.gouv.fr",
            "sample_10_rows": sample_json,
            "dagster/row_count": result["rows"],
            "dagster/table_name": f"{BRONZE_SCHEMA}.caracteristiques",
            "dagster/column_schema": table_schema,
        },
    )


@asset(
    group_name="bronze",
    description="Ingestion des usagers (victimes) depuis data.gouv.fr (2019-2024)",
    compute_kind="duckdb",
)
def usagers(context: AssetExecutionContext) -> Output[dict]:
    """
    Ingère les données usagers dans bronze.usagers.

    Source: data.gouv.fr CSV union 2019-2024
    Table: bronze.usagers
    """
    context.log.info("🔄 Ingestion usagers...")
    result = ingest_usagers()
    context.log.info(f"✅ Ingéré {result['rows']} lignes usagers")

    sample_json, table_schema = _build_table_metadata("usagers")

    return Output(
        result,
        metadata={
            "rows": result["rows"],
            "table": "bronze.usagers",
            "source": "data.gouv.fr",
            "sample_10_rows": sample_json,
            "dagster/row_count": result["rows"],
            "dagster/table_name": f"{BRONZE_SCHEMA}.usagers",
            "dagster/column_schema": table_schema,
        },
    )


@asset(
    group_name="bronze",
    description="Création de bronze.accidents_nc (filtré dep=988, Nouvelle-Calédonie)",
    compute_kind="duckdb",
    deps=["caracteristiques", "usagers"],
)
def accidents_nc(context: AssetExecutionContext) -> Output[dict]:
    """
    Crée la table accidents_nc en filtrant dep=988 et nettoyant les coordonnées.

    Transformations:
    - Filtre département 988 (Nouvelle-Calédonie)
    - Parse datetime français (jour/mois/an hrmn)
    - Nettoie coordonnées (trim, replace virgules)
    - Cast types appropriés

    Table: bronze.accidents_nc
    """
    context.log.info("🔄 Création accidents_nc (dep=988)...")
    result = create_accidents_nc()
    context.log.info(f"✅ Créé {result['rows']} accidents NC")

    sample_json, table_schema = _build_table_metadata("accidents_nc")

    return Output(
        result,
        metadata={
            "rows": result["rows"],
            "table": "bronze.accidents_nc",
            "department": "988",
            "sample_10_rows": sample_json,
            "dagster/row_count": result["rows"],
            "dagster/table_name": f"{BRONZE_SCHEMA}.accidents_nc",
            "dagster/column_schema": table_schema,
        },
    )
