"""Asset Dagster pour le blending d'ensemble ML.

Cet asset calcule les probabilités en combinant les sorties des modèles de
base et sauvegarde les prédictions finales dans `blend_preds.csv`.
Il ne sérialise pas un objet "modèle" unique.
"""
from dagster import asset
from dagster_pipeline.assets.blending_utils import predict_blend
from dagster_pipeline.assets.utils import save_predictions
from pipeline.stage_datasets import FEATURE_COLUMNS
from pipeline.config import GOLD_SCHEMA
from src.accidents.ducklake import get_client


@asset(group_name="gold", deps=["tune_catboost", "tune_xgboost", "tune_mlp"])
def blend_model():
    conn = get_client().conn
    test_df = conn.execute(f"SELECT * FROM {GOLD_SCHEMA}.test").df()
    X_test = test_df[FEATURE_COLUMNS]
    X_test_np = X_test.values  # conversion explicite en numpy array
    model_paths = ["catboost_model.pkl", "xgboost_model.pkl", "mlp_model.pkl"]
    proba_blend = predict_blend(X_test_np, model_paths)
    save_predictions(proba_blend, "blend_preds.csv")
    return "blend_preds.csv"
