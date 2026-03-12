"""Asset Dagster pour entraînement CatBoost."""

import json

import joblib
import matplotlib.pyplot as plt
import mlflow.catboost
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from dagster import AssetExecutionContext, asset
from mlflow.models import infer_signature
from optuna.integration import CatBoostPruningCallback
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.assets.gold.datasets import FEATURE_COLUMNS
from src.assets.gold.schema import GOLD_SCHEMA
from src.ducklake import get_client
from src.utils.models import save_model, save_predictions

# Features catégoriques pour CatBoost
CAT_FEATURES = [
    "geo_cluster",
    "road_type",
    "lit",
    "surface",
    "oneway",
    "atm",
]


@asset(group_name="gold", deps=["ml_datasets"], required_resource_keys={"mlflow"})
def tune_catboost(context: AssetExecutionContext):
    registered_model_name = "accidents_catboost"

    conn = get_client().conn
    train_df = conn.execute(f"SELECT * FROM {GOLD_SCHEMA}.train").df()
    test_df = conn.execute(f"SELECT * FROM {GOLD_SCHEMA}.test").df()
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["target"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["target"]

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    def objective(trial):
        cb_params = {
            "iterations": trial.suggest_int("iterations", 150, 500),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "auto_class_weights": "Balanced",
            "eval_metric": "AUC",
            "random_state": 42,
            "verbose": False,
        }
        model = CatBoostClassifier(**cb_params)
        model.fit(
            X_fit,
            y_fit,
            eval_set=(X_val, y_val),
            cat_features=CAT_FEATURES,
            callbacks=[CatBoostPruningCallback(trial, "AUC")],
            early_stopping_rounds=50,
            verbose=False,
        )
        preds = model.predict(X_val)
        return f1_score(y_val, preds, pos_label=1)

    study = optuna.create_study(
        direction="maximize",
        study_name="CatBoost",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    best_model = CatBoostClassifier(
        **study.best_params,
        auto_class_weights="Balanced",
        eval_metric="AUC",
        random_state=42,
        verbose=False,
    )
    best_model.fit(X_train, y_train, cat_features=CAT_FEATURES)
    save_model(best_model, "catboost_model.pkl")
    preds = best_model.predict_proba(X_test)
    save_predictions(preds, "catboost_preds.csv")

    # Log dans MLflow via resource Dagster
    mlflow_resource = context.resources.mlflow
    mlflow_resource.log_params({f"cb_{k}": v for k, v in study.best_params.items()})

    # Enregistrer le modèle dans le Model Registry (source de vérité cible)
    signature = infer_signature(X_train, best_model.predict_proba(X_train))
    model_info = mlflow.catboost.log_model(
        cb_model=best_model,
        artifact_path="model",
        registered_model_name=registered_model_name,
        signature=signature,
    )
    mlflow_resource.log_param("cb_registered_model_name", registered_model_name)
    mlflow_resource.log_param("cb_registered_model_uri", f"models:/{registered_model_name}/latest")
    mlflow_resource.log_param("cb_registered_model_source_uri", model_info.model_uri)

    y_proba = best_model.predict_proba(X_test)[:, 1]

    # ========== THRESHOLD TUNING (Precision-Recall Curve) ==========
    context.log.info("🎯 Recherche du seuil optimal...")
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    # Calcul F1-Score pour chaque seuil
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_threshold_idx]

    context.log.info(f"✅ Seuil optimal trouvé: {best_threshold:.4f} (F1={best_f1:.4f})")

    # Sauvegarder le seuil avec le modèle

    joblib.dump({"threshold": best_threshold}, "catboost_threshold.pkl")
    mlflow_resource.log_artifact("catboost_threshold.pkl")

    # Métriques avec seuil par défaut (0.5)
    y_pred_default = best_model.predict(X_test)
    recall_default = recall_score(y_test, y_pred_default)
    precision_default = precision_score(y_test, y_pred_default)
    f1_default = f1_score(y_test, y_pred_default)

    # Métriques avec seuil optimal
    y_pred_tuned = (y_proba >= best_threshold).astype(int)
    recall_tuned = recall_score(y_test, y_pred_tuned)
    precision_tuned = precision_score(y_test, y_pred_tuned)
    f1_tuned = f1_score(y_test, y_pred_tuned)

    auc_roc = roc_auc_score(y_test, y_proba)

    # Log métriques (threshold=0.5)
    mlflow_resource.log_metric("recall_default", recall_default)
    mlflow_resource.log_metric("precision_default", precision_default)
    mlflow_resource.log_metric("f1_default", f1_default)

    # Log métriques (threshold tuned)
    mlflow_resource.log_metric("recall_tuned", recall_tuned)
    mlflow_resource.log_metric("precision_tuned", precision_tuned)
    mlflow_resource.log_metric("f1_tuned", f1_tuned)
    mlflow_resource.log_metric("best_threshold", best_threshold)

    # Plot Precision-Recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, label="Precision-Recall curve")
    plt.scatter(
        [recalls[best_threshold_idx]],
        [precisions[best_threshold_idx]],
        color="red",
        s=100,
        zorder=5,
        label=f"Optimal (t={best_threshold:.3f}, F1={best_f1:.3f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - CatBoost")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("precision_recall_curve.png", dpi=150, bbox_inches="tight")
    mlflow_resource.log_artifact("precision_recall_curve.png")
    plt.close()

    # Feature Importance
    feature_importances = best_model.get_feature_importance()
    feature_names = FEATURE_COLUMNS
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": feature_importances}
    ).sort_values("importance", ascending=False)

    # Log top 10 comme métriques
    for _idx, row in importance_df.head(10).iterrows():
        mlflow_resource.log_metric(f"importance_{row['feature']}", row["importance"])

    # Créer et logger le plot
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df["feature"].head(15), importance_df["importance"].head(15))
    plt.xlabel("Importance")
    plt.title("CatBoost Feature Importance (Top 15)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("feature_importance_catboost.png", dpi=150, bbox_inches="tight")
    mlflow_resource.log_artifact("feature_importance_catboost.png")
    plt.close()

    # Logger le CSV complet
    importance_df.to_csv("feature_importance_catboost.csv", index=False)
    mlflow_resource.log_artifact("feature_importance_catboost.csv")

    # ========== SHAP EXPLICABILITY ==========
    context.log.info("🔍 Calcul SHAP values pour explicabilité...")
    try:
        import shap

        # TreeExplainer optimisé pour CatBoost
        explainer = shap.TreeExplainer(best_model)

        # Calculer SHAP values sur un échantillon du test (500 points max pour perf)
        sample_size = min(500, len(X_test))
        X_test_sample = X_test.sample(n=sample_size, random_state=42)
        shap_values = explainer.shap_values(X_test_sample)

        # Summary Plot (beeswarm)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_sample, feature_names=FEATURE_COLUMNS, show=False)
        plt.tight_layout()
        plt.savefig("shap_summary_plot.png", dpi=150, bbox_inches="tight")
        mlflow_resource.log_artifact("shap_summary_plot.png")
        plt.close()

        # Bar Plot (importance moyenne absolue)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_test_sample, feature_names=FEATURE_COLUMNS, plot_type="bar", show=False
        )
        plt.tight_layout()
        plt.savefig("shap_bar_plot.png", dpi=150, bbox_inches="tight")
        mlflow_resource.log_artifact("shap_bar_plot.png")
        plt.close()

        context.log.info("✅ SHAP plots générés et loggés")

    except ImportError:
        context.log.warning("⚠️  SHAP non installé, skipped (pip install shap)")
    except Exception as e:
        context.log.warning(f"⚠️  Erreur SHAP: {e}")

    mlflow_resource.log_artifact("catboost_model.pkl")
    mlflow_resource.log_artifact("catboost_preds.csv")

    # Sauvegarder métriques dans un fichier JSON pour blend_model
    catboost_metrics = {
        "registered_model_name": registered_model_name,
        "model_uri": f"models:/{registered_model_name}/latest",
        "model_path": "catboost_model.pkl",
        "preds_path": "catboost_preds.csv",
        "threshold": best_threshold,
        "auc_roc": auc_roc,
        "recall_tuned": recall_tuned,
        "precision_tuned": precision_tuned,
        "f1_tuned": f1_tuned,
    }
    with open("catboost_metrics.json", "w") as f:
        json.dump(catboost_metrics, f)
    mlflow_resource.log_artifact("catboost_metrics.json")

    return catboost_metrics
