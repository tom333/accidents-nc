import json

import mlflow
import mlflow.xgboost
import optuna
from dagster import AssetExecutionContext, asset
from mlflow.models import infer_signature
from sklearn.metrics import auc as auc_metric
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.assets.gold.datasets import FEATURE_COLUMNS
from src.assets.gold.schema import GOLD_SCHEMA
from src.ducklake import get_client
from src.utils.models import save_model, save_predictions


@asset(group_name="gold", deps=["ml_datasets"], required_resource_keys={"mlflow"})
def tune_xgboost(context: AssetExecutionContext):
    # Important: Désactiver l'autolog global pour éviter qu'XGBoost n'enregistre
    # automatiquement les params de chaque itération d'Optuna vers le même Run MLFlow.
    mlflow.xgboost.autolog(disable=True)
    mlflow.autolog(disable=True)

    registered_model_name = "accidents_xgboost"

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

    scale_weight = (y_train == 0).sum() / (y_train == 1).sum()

    def objective(trial):
        xgb_params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": scale_weight,
            "eval_metric": "logloss",
            "early_stopping_rounds": 50,
            "random_state": 42,
            "verbosity": 0,
        }
        model = XGBClassifier(**xgb_params)
        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        return f1_score(y_val, preds, pos_label=1)

    study = optuna.create_study(
        direction="maximize",
        study_name="XGBoost",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    best_model = XGBClassifier(
        **study.best_params, scale_pos_weight=scale_weight, random_state=42, verbosity=0
    )
    best_model.fit(X_train, y_train)
    save_model(best_model, "xgboost_model.pkl")
    preds = best_model.predict_proba(X_test)
    save_predictions(preds, "xgboost_preds.csv")

    # Log dans MLflow via resource Dagster
    mlflow_resource = context.resources.mlflow
    mlflow_resource.log_params({f"xgb_{k}": v for k, v in study.best_params.items()})

    signature = infer_signature(X_train, best_model.predict_proba(X_train))
    model_info = mlflow.xgboost.log_model(
        xgb_model=best_model,
        artifact_path="model",
        registered_model_name=registered_model_name,
        signature=signature,
    )
    mlflow_resource.log_param("xgb_registered_model_name", registered_model_name)
    mlflow_resource.log_param("xgb_registered_model_uri", f"models:/{registered_model_name}/latest")
    mlflow_resource.log_param("xgb_registered_model_source_uri", model_info.model_uri)

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_proba)
    auc_val = auc_metric([0, 1], [recall, precision])

    mlflow_resource.log_metric("recall", recall)
    mlflow_resource.log_metric("precision", precision)
    mlflow_resource.log_metric("f1", f1)
    mlflow_resource.log_metric("auc", auc_val)
    mlflow_resource.log_metric("auc_roc", auc_roc)
    mlflow_resource.log_artifact("xgboost_model.pkl")
    mlflow_resource.log_artifact("xgboost_preds.csv")

    xgboost_metrics = {
        "registered_model_name": registered_model_name,
        "model_uri": f"models:/{registered_model_name}/latest",
        "model_path": "xgboost_model.pkl",
        "preds_path": "xgboost_preds.csv",
        "auc_roc": auc_roc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }
    with open("xgboost_metrics.json", "w") as f:
        json.dump(xgboost_metrics, f)
    mlflow_resource.log_artifact("xgboost_metrics.json")

    return xgboost_metrics
