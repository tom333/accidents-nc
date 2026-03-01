"""Asset Dagster pour l'évaluation du modèle d'ensemble (blending)."""
from dagster import asset
from dagster_pipeline.assets.blending_utils import predict_blend
from dagster_pipeline.assets.utils import load_predictions
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
from pipeline.stage_datasets import FEATURE_COLUMNS
from pipeline.config import GOLD_SCHEMA
from src.accidents.ducklake import get_client

@asset(group_name="gold", deps=["blend_model"])
def eval_blend():
    # Charger X_test et y_test depuis DuckLake
    conn = get_client().conn
    test_df = conn.execute(f"SELECT * FROM {GOLD_SCHEMA}.test").df()
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df['target'].values
    # Chemins des modèles
    model_paths = ["catboost_model.pkl", "xgboost_model.pkl", "mlp_model.pkl"]
    # Prédiction blending
    proba_blend = predict_blend(X_test, model_paths)
    preds_blend = (proba_blend > 0.5).astype(int)
    # Calcul des métriques
    recall = recall_score(y_test, preds_blend)
    precision = precision_score(y_test, preds_blend)
    f1 = f1_score(y_test, preds_blend)
    auc = roc_auc_score(y_test, proba_blend)
    metrics = {
        "recall": recall,
        "precision": precision,
        "f1_score": f1,
        "auc_roc": auc
    }
    # Logger les résultats (ex: MLflow)
    # ...existing code...
    return metrics
