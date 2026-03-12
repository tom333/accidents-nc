"""Asset Dagster pour le blending d'ensemble ML.

Cet asset calcule les probabilités en combinant les sorties des modèles de
base et sauvegarde les prédictions finales dans `blend_preds.csv`.

Il utilise `deps=` (et non des arguments de fonction) pour éviter la dépendance
à l'I/O Manager de Dagster : il peut ainsi être matérialisé seul, sans avoir
à relancer tous les assets amont.

Prérequis sur disque :
  - catboost_model.pkl / xgboost_model.pkl / mlp_model.pkl
  - catboost_metrics.json / xgboost_metrics.json / mlp_metrics.json
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dagster import AssetExecutionContext, asset
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.assets.gold.blending_utils import predict_blend
from src.assets.gold.datasets import FEATURE_COLUMNS
from src.assets.gold.schema import GOLD_SCHEMA
from src.ducklake import get_client
from src.utils.models import save_predictions


def _load_metrics(path: str) -> dict:
    """Charge les métriques d'un modèle individuel depuis son JSON."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


@asset(
    group_name="gold",
    required_resource_keys={"mlflow"},
    deps=["tune_catboost", "tune_xgboost", "tune_mlp"],  # ordering only, no IO
)
def blend_model(context: AssetExecutionContext):
    """Blending avec évaluation complète et comparaison MLflow.

    Utilise deps= (pas d'arguments de fonction) pour être matérialisable
    indépendamment sans relancer les assets amont.
    Les modèles .pkl et les métriques .json sont lus directement sur le disque.
    """
    # ─── Lecture des métriques individuelles (depuis les JSON) ───
    cat_metrics = _load_metrics("catboost_metrics.json")
    xgb_metrics = _load_metrics("xgboost_metrics.json")
    mlp_metrics = _load_metrics("mlp_metrics.json")

    model_paths = [
        cat_metrics.get("model_path", "catboost_model.pkl"),
        xgb_metrics.get("model_path", "xgboost_model.pkl"),
        mlp_metrics.get("model_path", "mlp_model.pkl"),
    ]
    for p in model_paths:
        if not Path(p).exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {p}. Matérialisez d'abord les assets amont."
            )

    # ─── Chargement du test set ───
    conn = get_client().conn
    test_df = conn.execute(f"SELECT * FROM {GOLD_SCHEMA}.test").df()
    y_test = test_df["target"].values
    X_test = test_df[FEATURE_COLUMNS]

    # ─── Prédictions blend ───
    proba_blend = predict_blend(X_test, model_paths)
    save_predictions(proba_blend, "blend_preds.csv")

    # ─── Seuil optimal (Precision-Recall) ───
    context.log.info("🎯 Recherche du seuil optimal pour le blend...")
    precisions, recalls, thresholds = precision_recall_curve(y_test, proba_blend)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    best_f1 = float(f1_scores[best_idx])
    context.log.info(f"✅ Seuil optimal blend: {best_threshold:.4f} (F1={best_f1:.4f})")

    # ─── Métriques blend ───
    y_pred_blend = (proba_blend >= best_threshold).astype(int)
    blend_recall = recall_score(y_test, y_pred_blend)
    blend_precision = precision_score(y_test, y_pred_blend)
    blend_f1 = f1_score(y_test, y_pred_blend)
    blend_auc = roc_auc_score(y_test, proba_blend)
    context.log.info(
        f"📊 Blend — Recall: {blend_recall:.3f} | Precision: {blend_precision:.3f} | "
        f"F1: {blend_f1:.3f} | AUC-ROC: {blend_auc:.3f}"
    )

    # ─── MLflow logging ───
    mlflow = context.resources.mlflow
    mlflow.log_param("weights", "catboost=0.4, xgboost=0.4, mlp=0.2")
    mlflow.log_param("best_threshold", best_threshold)
    mlflow.log_metric("recall", blend_recall)
    mlflow.log_metric("precision", blend_precision)
    mlflow.log_metric("f1", blend_f1)
    mlflow.log_metric("auc_roc", blend_auc)
    # AUC individuels pour comparaison rapide dans la même run
    for name, m in [("catboost", cat_metrics), ("xgboost", xgb_metrics), ("mlp", mlp_metrics)]:
        if m.get("auc_roc") is not None:
            mlflow.log_metric(f"{name}_auc_roc", m["auc_roc"])
        if m.get("model_uri"):
            mlflow.log_param(f"{name}_model_uri", m["model_uri"])

    # ─── Plot : Precision-Recall du blend ───
    plt.figure(figsize=(9, 6))
    plt.plot(recalls, precisions, color="#4C72B0", lw=2, label="Precision-Recall (Blend)")
    plt.scatter(
        [recalls[best_idx]],
        [precisions[best_idx]],
        color="red",
        s=120,
        zorder=5,
        label=f"Seuil optimal (t={best_threshold:.3f}, F1={best_f1:.3f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Ensemble Blend")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("blend_precision_recall_curve.png", dpi=150, bbox_inches="tight")
    mlflow.log_artifact("blend_precision_recall_curve.png")
    plt.close()

    # ─── Plot : Comparaison AUC + métriques blend ───
    cat_auc = cat_metrics.get("auc_roc") or 0.0
    xgb_auc = xgb_metrics.get("auc_roc") or 0.0
    mlp_auc = mlp_metrics.get("auc_roc") or 0.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Bar chart AUC : CatBoost / XGBoost / MLP / Blend
    labels = ["CatBoost", "XGBoost", "MLP", "Blend"]
    aucs = [cat_auc, xgb_auc, mlp_auc, blend_auc]
    colors = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]
    bars = axes[0].bar(labels, aucs, color=colors, width=0.5, edgecolor="white")
    axes[0].set_ylim(max(0, min(aucs) - 0.02), 1.0)
    axes[0].set_ylabel("AUC-ROC")
    axes[0].set_title("Comparaison AUC-ROC")
    for bar, val in zip(bars, aucs, strict=False):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Bar chart Recall / Precision / F1 du blend avec seuil optimal
    metrics_names = ["Recall", "Precision", "F1"]
    metrics_vals = [blend_recall, blend_precision, blend_f1]
    metric_colors = ["#4878CF", "#6ACC65", "#D65F5F"]
    bars2 = axes[1].bar(
        metrics_names, metrics_vals, color=metric_colors, width=0.4, edgecolor="white"
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Score")
    axes[1].set_title(f"Blend — Métriques (seuil={best_threshold:.3f})")
    for bar, val in zip(bars2, metrics_vals, strict=False):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    fig.suptitle(
        "Évaluation Ensemble Blending vs Modèles Individuels", fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig("blend_comparison.png", dpi=150, bbox_inches="tight")
    mlflow.log_artifact("blend_comparison.png")
    plt.close(fig)

    mlflow.log_artifact("blend_preds.csv")
    context.log.info("✅ Plots blend générés et loggés dans MLflow.")

    # ─── Feature Importance globale du blend (Permutation Importance) ───
    context.log.info("🔍 Calcul de la Permutation Importance sur le blend...")
    try:
        import pandas as pd
        from sklearn.inspection import permutation_importance

        from src.assets.gold.blending_utils import BlendingEnsembleWrapper
        from src.utils.models import load_model

        # Charger les modèles en mémoire
        cat_model = load_model(model_paths[0])
        xgb_model = load_model(model_paths[1])
        mlp_model = load_model(model_paths[2])

        ensemble_model = BlendingEnsembleWrapper(
            cat_model=cat_model,
            xgb_model=xgb_model,
            mlp_model=mlp_model,
            mlp_weights=[0.4, 0.4, 0.2],
            threshold=best_threshold,
        )

        # On utilise un échantillon pour des raisons de performance
        sample_size = min(2000, len(X_test))
        X_test_sample = X_test.sample(n=sample_size, random_state=42)
        y_test_sample = y_test[X_test_sample.index]

        # Permutation importance
        result = permutation_importance(
            ensemble_model,
            X_test_sample,
            y_test_sample,
            n_repeats=5,
            random_state=42,
            scoring="roc_auc",
            n_jobs=-1,
        )

        importance_df = pd.DataFrame(
            {"feature": FEATURE_COLUMNS, "importance": result.importances_mean}
        ).sort_values("importance", ascending=False)

        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(
            importance_df["feature"].head(15), importance_df["importance"].head(15), color="#B47CC7"
        )
        plt.xlabel("Permutation Importance (Mean AUC decrease)")
        plt.title("Blend Feature Importance (Top 15)")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig("blend_feature_importance.png", dpi=150, bbox_inches="tight")
        mlflow.log_artifact("blend_feature_importance.png")
        plt.close()

        importance_df.to_csv("blend_feature_importance.csv", index=False)
        mlflow.log_artifact("blend_feature_importance.csv")
        context.log.info("✅ Feature importance du blend générée et loggée.")

    except Exception as e:
        context.log.error(f"⚠️ Erreur lors du calcul de la permutation importance: {e}")

    # Sauvegarder métriques blend pour usage aval
    blend_metrics = {
        "preds_path": "blend_preds.csv",
        "threshold": best_threshold,
        "weights": {"catboost": 0.4, "xgboost": 0.4, "mlp": 0.2},
        "model_versions": {
            "catboost": cat_metrics.get("model_uri"),
            "xgboost": xgb_metrics.get("model_uri"),
            "mlp": mlp_metrics.get("model_uri"),
        },
        "recall": blend_recall,
        "precision": blend_precision,
        "f1": blend_f1,
        "auc_roc": blend_auc,
    }
    with open("blend_metrics.json", "w") as f:
        json.dump(blend_metrics, f)
    mlflow.log_artifact("blend_metrics.json")

    return blend_metrics
