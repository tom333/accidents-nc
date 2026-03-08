"""Silver layer assets - Feature engineering."""

from dagster import AssetExecutionContext, Output, asset

from src.assets.silver.features import build_feature_store


@asset(
    group_name="silver",
    description="Construction du feature store avec OSM, négatifs et features temporelles",
    compute_kind="python",
    deps=["accidents_nc"],
)
def full_dataset(context: AssetExecutionContext) -> Output[dict]:
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

    return Output(
        result,
        metadata={
            "rows": result["rows"],
            "positives": result["positives"],
            "negatives": result["negatives"],
            "positive_rate": f"{positive_pct:.2f}%",
            "table": "silver.full_dataset",
            "features_count": 18,
        },
    )
