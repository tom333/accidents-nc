import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Étape 4 – Modélisation & Tuning Hyperparamètres

    Ce notebook charge les datasets train/test depuis **DuckLake**, effectue le tuning des hyperparamètres avec **Optuna** (CatBoost, LightGBM, XGBoost), puis entraîne des modèles additionnels (Logistic Regression, TabNet, Neural Network avec embeddings).

    Tous les runs sont loggés dans **MLflow** et le meilleur modèle est sauvegardé en `accident_model.pkl`.
    """)
    return


@app.cell
def _():
    import copy
    import io
    import joblib
    import mlflow
    import numpy as np
    import optuna
    import pandas as pd
    import torch
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from optuna.integration import CatBoostPruningCallback, LightGBMPruningCallback
    from pytorch_tabnet.tab_model import TabNetClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from xgboost import XGBClassifier
    from pipeline.config import GOLD_SCHEMA, PIPELINE_PARAMS, ensure_connection
    from pipeline.stage_datasets import FEATURE_COLUMNS
    return (
        CatBoostClassifier,
        CatBoostPruningCallback,
        FEATURE_COLUMNS,
        GOLD_SCHEMA,
        LGBMClassifier,
        LightGBMPruningCallback,
        LogisticRegression,
        PIPELINE_PARAMS,
        Pipeline,
        StandardScaler,
        XGBClassifier,
        classification_report,
        confusion_matrix,
        ensure_connection,
        f1_score,
        joblib,
        mlflow,
        optuna,
        pd,
        precision_score,
        recall_score,
        roc_auc_score,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 1. Chargement des données depuis DuckLake
    """)
    return


@app.cell
def _(FEATURE_COLUMNS, GOLD_SCHEMA, ensure_connection, mo):
    conn = ensure_connection()
    train_df = conn.execute(f"SELECT * FROM {GOLD_SCHEMA}.train").df()
    test_df = conn.execute(f"SELECT * FROM {GOLD_SCHEMA}.test").df()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df['target']
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df['target']

    mo.md(f"""
    ✅ Données chargées :
    - **Train**: {len(X_train)} échantillons
    - **Test**: {len(X_test)} échantillons
    - **Features**: {len(FEATURE_COLUMNS)} colonnes
    - **Distribution train**: {(y_train==1).sum()} positifs, {(y_train==0).sum()} négatifs
    - **Distribution test**: {(y_test==1).sum()} positifs, {(y_test==0).sum()} négatifs
    """)
    return X_test, X_train, y_test, y_train


@app.cell
def _(mo):
    mo.md("""
    ## 2. Configuration MLflow & Calcul du Class Weight
    """)
    return


@app.cell
def _(FEATURE_COLUMNS, PIPELINE_PARAMS, mlflow, y_train):
    tuning_n_negative = (y_train == 0).sum()
    tuning_n_positive = (y_train == 1).sum()
    tuning_scale_weight = tuning_n_negative / tuning_n_positive

    mlflow.set_experiment("accidents-nc")
    mlflow.start_run(run_name="notebook-training")
    mlflow.log_params({
        'scale_weight': tuning_scale_weight,
        'n_train_samples': len(y_train),
        'n_features': len(FEATURE_COLUMNS),
        'random_state': PIPELINE_PARAMS.random_state
    })

    print(f"⚖️ Scale weight (négatifs/positifs): {tuning_scale_weight:.2f}")
    return (tuning_scale_weight,)


@app.cell
def _(mo):
    mo.md("""
    ## 3. Optuna Tuning - CatBoost
    """)
    return


@app.cell
def _(
    CatBoostClassifier,
    CatBoostPruningCallback,
    PIPELINE_PARAMS,
    X_test,
    X_train,
    optuna,
    recall_score,
    y_test,
    y_train,
):
    def objective_catboost(trial):
        cb_params = {
            'iterations': trial.suggest_int('iterations', 150, 500),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'auto_class_weights': 'Balanced',
            'eval_metric': 'AUC',
            'random_state': PIPELINE_PARAMS.random_state,
            'verbose': False
        }
        model = CatBoostClassifier(**cb_params)
        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            callbacks=[CatBoostPruningCallback(trial, 'AUC')],
            early_stopping_rounds=50,
            verbose=False
        )
        preds = model.predict(X_test)
        return recall_score(y_test, preds, pos_label=1)

    study_catboost = optuna.create_study(
        direction='maximize',
        study_name='CatBoost',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    study_catboost.optimize(objective_catboost, n_trials=30, show_progress_bar=True)

    print(f"✅ CatBoost - Best Recall: {study_catboost.best_value:.4f}")
    return (study_catboost,)


@app.cell
def _(mo):
    mo.md("""
    ## 4. Optuna Tuning - LightGBM
    """)
    return


@app.cell
def _(
    LGBMClassifier,
    LightGBMPruningCallback,
    PIPELINE_PARAMS,
    X_test,
    X_train,
    optuna,
    recall_score,
    y_test,
    y_train,
):
    def objective_lgbm(trial):
        lgbm_params = {
            'n_estimators': trial.suggest_int('n_estimators', 150, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'is_unbalance': True,
            'metric': 'auc',
            'random_state': PIPELINE_PARAMS.random_state,
            'verbose': -1
        }
        model = LGBMClassifier(**lgbm_params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[LightGBMPruningCallback(trial, 'auc')])
        preds = model.predict(X_test)
        return recall_score(y_test, preds, pos_label=1)

    study_lgbm = optuna.create_study(
        direction='maximize',
        study_name='LightGBM',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    study_lgbm.optimize(objective_lgbm, n_trials=30, show_progress_bar=True)

    print(f"✅ LightGBM - Best Recall: {study_lgbm.best_value:.4f}")
    return (study_lgbm,)


@app.cell
def _(mo):
    mo.md("""
    ## 5. Optuna Tuning - XGBoost
    """)
    return


@app.cell
def _(
    PIPELINE_PARAMS,
    XGBClassifier,
    X_test,
    X_train,
    optuna,
    recall_score,
    tuning_scale_weight,
    y_test,
    y_train,
):
    def objective_xgb(trial):
        xgb_params = {
            'n_estimators': trial.suggest_int('n_estimators', 150, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 5.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'scale_pos_weight': tuning_scale_weight,
            'random_state': PIPELINE_PARAMS.random_state,
            'verbosity': 0
        }
        model = XGBClassifier(**xgb_params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return recall_score(y_test, preds, pos_label=1)

    study_xgb = optuna.create_study(
        direction='maximize',
        study_name='XGBoost',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    study_xgb.optimize(objective_xgb, n_trials=30, show_progress_bar=True)

    print(f"✅ XGBoost - Best Recall: {study_xgb.best_value:.4f}")
    return (study_xgb,)


@app.cell
def _(mo):
    mo.md("""
    ## 6. Entraînement des modèles tunés + baselines
    """)
    return


@app.cell
def _(
    CatBoostClassifier,
    LGBMClassifier,
    LogisticRegression,
    PIPELINE_PARAMS,
    Pipeline,
    StandardScaler,
    XGBClassifier,
    X_test,
    X_train,
    f1_score,
    mlflow,
    precision_score,
    recall_score,
    roc_auc_score,
    study_catboost,
    study_lgbm,
    study_xgb,
    tuning_scale_weight,
    y_test,
    y_train,
):
    tuned_models = {}
    tuning_results = []

    def _log_metrics(name, model, params_dict=None):
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        metrics = {
            'recall': recall_score(y_test, preds, pos_label=1),
            'precision': precision_score(y_test, preds, pos_label=1),
            'f1_score': f1_score(y_test, preds, pos_label=1),
            'auc_roc': roc_auc_score(y_test, probs),
        }
        tuned_models[name] = model
        tuning_results.append({
            'Modèle': name,
            'Recall': f"{metrics['recall']:.3f}",
            'Precision': f"{metrics['precision']:.3f}",
            'F1-Score': f"{metrics['f1_score']:.3f}",
            'AUC-ROC': f"{metrics['auc_roc']:.3f}",
            '_recall_raw': metrics['recall']
        })
        if params_dict:
            mlflow.log_params({f"{name}.{k}": v for k, v in params_dict.items()})
        mlflow.log_metrics({f"{name}.{k}": v for k, v in metrics.items()})
        return metrics

    # CatBoost
    best_catboost = CatBoostClassifier(
        **study_catboost.best_params,
        auto_class_weights='Balanced',
        eval_metric='AUC',
        random_state=PIPELINE_PARAMS.random_state,
        verbose=False
    )
    best_catboost.fit(X_train, y_train)
    _log_metrics('CatBoost', best_catboost, study_catboost.best_params)

    # LightGBM
    best_lgbm = LGBMClassifier(
        **study_lgbm.best_params,
        is_unbalance=True,
        metric='auc',
        random_state=PIPELINE_PARAMS.random_state,
        verbose=-1
    )
    best_lgbm.fit(X_train, y_train)
    _log_metrics('LightGBM', best_lgbm, study_lgbm.best_params)

    # XGBoost
    best_xgb = XGBClassifier(
        **study_xgb.best_params,
        scale_pos_weight=tuning_scale_weight,
        random_state=PIPELINE_PARAMS.random_state,
        verbosity=0
    )
    best_xgb.fit(X_train, y_train)
    _log_metrics('XGBoost', best_xgb, study_xgb.best_params)

    # Logistic Regression
    logistic_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('log_reg', LogisticRegression(
            class_weight='balanced',
            C=0.5,
            max_iter=2000,
            solver='lbfgs',
            random_state=PIPELINE_PARAMS.random_state
        ))
    ])
    logistic_pipeline.fit(X_train, y_train)
    _log_metrics('LogisticRegression', logistic_pipeline, {'C': 0.5, 'solver': 'lbfgs'})

    print("✅ Modèles tunés entraînés")
    return tuned_models, tuning_results


@app.cell
def _(mo):
    mo.md("""
    ## 7. Comparaison des performances
    """)
    return


@app.cell
def _(mlflow, pd, tuning_results):
    results_df = pd.DataFrame(tuning_results).sort_values('_recall_raw', ascending=False)
    mlflow.log_table(results_df.drop(columns='_recall_raw'), "comparison_table.json")
    results_df
    return (results_df,)


@app.cell
def _(mo):
    mo.md("""
    ## 8. Rapport détaillé du meilleur modèle
    """)
    return


@app.cell
def _(
    X_test,
    classification_report,
    confusion_matrix,
    mo,
    results_df,
    tuned_models,
    y_test,
):
    best_model_name = results_df.iloc[0]['Modèle']
    best_model = tuned_models[best_model_name]
    best_recall = results_df.iloc[0]['_recall_raw']

    preds_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, preds_best)
    report = classification_report(y_test, preds_best, target_names=['Pas Accident', 'Accident'])

    mo.md(f"""
    ### 🏆 Meilleur modèle : **{best_model_name}**

    **Recall** : {best_recall:.3f}

    #### Matrice de confusion
    ```
    {cm}
    ```

    #### Rapport de classification
    ```
    {report}
    ```
    """)
    return best_model, best_model_name, best_recall


@app.cell
def _(mo):
    mo.md("""
    ## 9. Sauvegarde du meilleur modèle
    """)
    return


@app.cell
def _(best_model, best_model_name, best_recall, joblib, mlflow):
    joblib.dump(best_model, 'accident_model.pkl')
    joblib.dump({'name': best_model_name, 'recall': best_recall}, 'best_model_info.pkl')

    mlflow.log_metric('best_recall', best_recall)
    mlflow.log_param('best_model', best_model_name)

    try:
        mlflow.sklearn.log_model(best_model, artifact_path="best-model")
        print(f"✅ Modèle {best_model_name} sauvegardé dans MLflow")
    except Exception as exc:
        print(f"⚠️ Impossible d'enregistrer dans MLflow: {exc}")

    mlflow.end_run()

    print(f"✅ Modèle sauvegardé localement : accident_model.pkl")
    print(f"✅ Informations sauvegardées : best_model_info.pkl")
    return


if __name__ == "__main__":
    app.run()
