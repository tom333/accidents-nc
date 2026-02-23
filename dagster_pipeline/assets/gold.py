"""Gold layer assets - ML datasets."""

from dagster import asset, Output, AssetExecutionContext

from src.accidents.gold.datasets import build_datasets
from src.accidents.gold.training import train_models


@asset(
    group_name="gold",
    description="Création des datasets train/test ML avec encodage",
    compute_kind="sklearn",
    deps=["full_dataset"],
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
    
    train_pos_pct = (result['train_positives'] / result['train_rows']) * 100
    test_pos_pct = (result['test_positives'] / result['test_rows']) * 100
    
    context.log.info(
        f"✅ Datasets créés: "
        f"train={result['train_rows']} ({result['train_positives']} positifs [{train_pos_pct:.1f}%]), "
        f"test={result['test_rows']} ({result['test_positives']} positifs [{test_pos_pct:.1f}%])"
    )
    
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
            "s3_artifacts": "atm_encoder.pkl, features.pkl",
        },
    )


@asset(
    group_name="gold",
    description="Entrainement du modele ML (CatBoost) -> accident_model.pkl",
    compute_kind="catboost",
    deps=["ml_datasets"],
)
def ml_models(context: AssetExecutionContext) -> Output[dict]:
    """Entraine un modele CatBoost et stocke l'artefact sur S3."""
    context.log.info("🔄 Entrainement modele ML...")
    result = train_models()
    best_metrics = result.get("best_metrics", {})
    if best_metrics:
        context.log.info(
            "✅ Modele entraine: "
            f"auc={best_metrics.get('auc', 0.0):.4f}, "
            f"recall={best_metrics.get('recall', 0.0):.4f}, "
            f"precision={best_metrics.get('precision', 0.0):.4f}, "
            f"f1={best_metrics.get('f1', 0.0):.4f}"
        )

    return Output(
        result,
        metadata={
            "train_rows": result["train_rows"],
            "test_rows": result["test_rows"],
            "best_model": result.get("best_model", ""),
            "auc": f"{best_metrics.get('auc', 0.0):.4f}",
            "recall": f"{best_metrics.get('recall', 0.0):.4f}",
            "precision": f"{best_metrics.get('precision', 0.0):.4f}",
            "f1": f"{best_metrics.get('f1', 0.0):.4f}",
            "s3_artifact": "accident_model.pkl",
        },
    )
