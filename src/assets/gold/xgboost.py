import json

import optuna
from dagster import AssetExecutionContext, asset
from sklearn.metrics import auc as auc_metric
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from src.assets.gold.datasets import FEATURE_COLUMNS
from src.assets.gold.schema import GOLD_SCHEMA
from src.ducklake import get_client
from src.utils.models import save_model, save_predictions


@asset(group_name="gold", deps=["ml_datasets"], required_resource_keys={"mlflow"})
def tune_xgboost(context: AssetExecutionContext):
    conn = get_client().conn
    train_df = conn.execute(f"SELECT * FROM {GOLD_SCHEMA}.train").df()
    test_df = conn.execute(f"SELECT * FROM {GOLD_SCHEMA}.test").df()
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["target"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["target"]
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
            "random_state": 42,
            "verbosity": 0,
        }
        model = XGBClassifier(**xgb_params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return recall_score(y_test, preds, pos_label=1)

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
    mlflow = context.resources.mlflow
    mlflow.log_params(study.best_params)

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_proba)
    auc_val = auc_metric([0, 1], [recall, precision])

    mlflow.log_metric("recall", recall)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("auc", auc_val)
    mlflow.log_metric("auc_roc", auc_roc)
    mlflow.log_artifact("xgboost_model.pkl")
    mlflow.log_artifact("xgboost_preds.csv")

    xgboost_metrics = {
        "model_path": "xgboost_model.pkl",
        "preds_path": "xgboost_preds.csv",
        "auc_roc": auc_roc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }
    with open("xgboost_metrics.json", "w") as f:
        json.dump(xgboost_metrics, f)
    mlflow.log_artifact("xgboost_metrics.json")

    return xgboost_metrics
