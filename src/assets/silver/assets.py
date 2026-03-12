"""Silver layer assets - Feature engineering."""

import json

from dagster import AssetExecutionContext, Output, TableColumn, TableSchema, asset

from src.assets.silver.features import build_feature_store
from src.assets.silver.schema import SILVER_SCHEMA
from src.ducklake import get_client


@asset(
    group_name="silver",
    description="Construction du feature store avec OSM, négatifs et features temporelles",
    compute_kind="python",
    deps=["accidents_nc"],
)
def full_dataset(context: AssetExecutionContext) -> Output[dict[str, int]]:
    """
    Construit silver.full_dataset avec features enrichies.

    Pipeline:
    1. Charge accidents depuis bronze.accidents_nc
    2. Télécharge/cache routes OSM (30+ communes NC)
    3. Génère grille spatiale (buffer 200m, step 0.02°)
    4. Échantillonne négatifs (exclusion 300m, ratio 85% risque)
    5. Calcule features temporelles (weekend, rush hour, cyclique)
    6. Ajoute holidays NC et vacances scolaires

    Table: silver.full_dataset
    Features: 18 (lat, lon, hour, dayofweek, month, atm, is_weekend,
              is_rush_morning, is_rush_evening, is_night, hour_sin/cos,
              dayofweek_sin/cos, road_type, speed_limit, is_holiday,
              school_holidays, target)
    """
    context.log.info("🔄 Construction feature store...")
    result = build_feature_store()

    positive_pct = (result["positives"] / result["rows"]) * 100
    context.log.info(
        f"✅ Feature store créé: {result['rows']} lignes "
        f"({result['positives']} positifs [{positive_pct:.1f}%], "
        f"{result['negatives']} négatifs)"
    )

    # Charger un extrait des données pour les métadonnées
    client = get_client()
    conn = client.conn
    sample_df = conn.execute(f"SELECT * FROM {SILVER_SCHEMA}.full_dataset LIMIT 10").df()

    # Convertir en JSON string (Dagster requiert des types sérialisables)
    sample_json = json.dumps(sample_df.to_dict(orient="records"), default=str)

    # Metadata table/colonnes Dagster
    columns = [
        TableColumn(name=str(col), type=str(dtype)) for col, dtype in sample_df.dtypes.items()
    ]
    table_schema = TableSchema(columns=columns)

    return Output(
        result,
        metadata={
            "rows": result["rows"],
            "positives": result["positives"],
            "negatives": result["negatives"],
            "positive_rate": f"{positive_pct:.2f}%",
            "table": "silver.full_dataset",
            "features_count": 18,
            "sample_10_rows": sample_json,
            "dagster/row_count": result["rows"],
            "dagster/table_name": f"{SILVER_SCHEMA}.full_dataset",
            "dagster/column_schema": table_schema,
        },
    )
