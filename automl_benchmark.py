import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # 🤖 AutoML Benchmark avec AutoGluon

        Ce notebook compare automatiquement 15+ algorithmes ML pour trouver le meilleur modèle de prédiction d'accidents.

        **Fonctionnalités** :
        - Comparaison automatique de multiples algorithmes (LightGBM, XGBoost, CatBoost, RandomForest, KNN, etc.)
        - Optimisation hyperparamètres avec validation croisée
        - Stacking automatique des meilleurs modèles
        - Rapport détaillé avec leaderboard et feature importance

        **Durée estimée** : 30-60 minutes (selon `time_limit`)
        """
    )
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import polars as pl
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, f1_score, roc_auc_score
    import joblib
    import time
    return (
        LabelEncoder,
        classification_report,
        confusion_matrix,
        f1_score,
        joblib,
        np,
        pd,
        pl,
        precision_score,
        recall_score,
        roc_auc_score,
        time,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md("""## ⚙️ Configuration""")
    return


@app.cell
def _():
    CONFIG = {
        'time_limit': 1800,  # 30 minutes (ajuster selon besoins : 3600 = 1h)
        'preset': 'best_quality',  # Options : 'medium_quality', 'good_quality', 'best_quality', 'optimize_for_deployment'
        'eval_metric': 'recall',  # Priorité : détecter accidents
        'random_state': 42,
        'test_size': 0.1
    }
    return (CONFIG,)


@app.cell
def _(CONFIG, mo):
    mo.md(f"""
    **Paramètres AutoML** :
    - ⏱️ Temps limite : {CONFIG['time_limit'] // 60} minutes
    - 🎯 Métrique : {CONFIG['eval_metric']} (priorité détection accidents)
    - 🏆 Qualité : {CONFIG['preset']}
    - 🎲 Random state : {CONFIG['random_state']}
    """)
    return


@app.cell
def _(mo):
    mo.md("""## 📥 Chargement des Données""")
    return


@app.cell
def _():
    # Charger le dataset complet depuis accident_fetch_data.py
    import sys
    sys.path.append('.')
    
    try:
        full_dataset = joblib.load('full_dataset.pkl')
        print(f"✅ Dataset chargé depuis cache : {len(full_dataset):,} lignes")
    except FileNotFoundError:
        print("⚠️ Cache non trouvé. Exécutez accident_fetch_data.py d'abord pour générer full_dataset.pkl")
        print("💡 Ou utilisez ce code pour charger depuis les CSV...")
        raise
    return (full_dataset,)


@app.cell
def _(full_dataset, mo):
    mo.md(f"""
    **Dataset chargé** :
    - 📊 Lignes : {len(full_dataset):,}
    - 🎯 Target : `target` (0 = pas d'accident, 1 = accident)
    - 📈 Features : 24 (6 de base + 18 enrichies)
    """)
    
    full_dataset.head()
    return


@app.cell
def _(mo):
    mo.md("""## 🔧 Préparation des Features""")
    return


@app.cell
def _(CONFIG, LabelEncoder, full_dataset, train_test_split):
    # Encoder la feature catégorielle 'atm'
    le_automl = LabelEncoder()
    tmp_dataset_automl = full_dataset.to_pandas()
    tmp_dataset_automl['atm'] = le_automl.fit_transform(tmp_dataset_automl['atm'])
    
    # Liste complète des 24 features
    features_automl = [
        'latitude', 'longitude', 'hour', 'dayofweek', 'month', 'atm',
        'lat_hour', 'lon_hour', 'lat_dayofweek', 'lon_dayofweek',
        'is_weekend', 'is_rush_morning', 'is_rush_evening', 'is_night',
        'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
        'road_type', 'speed_limit', 'accident_density_5km', 'dist_to_noumea_km',
        'is_holiday', 'school_holidays'
    ]
    
    X_automl = tmp_dataset_automl[features_automl]
    y_automl = tmp_dataset_automl['target']
    
    # Split train/test
    X_train_automl, X_test_automl, y_train_automl, y_test_automl = train_test_split(
        X_automl, y_automl,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=y_automl
    )
    
    print(f"✅ Features préparées : {len(features_automl)} colonnes")
    print(f"📊 Train : {len(X_train_automl):,} | Test : {len(X_test_automl):,}")
    print(f"🎯 Distribution target train : {y_train_automl.value_counts().to_dict()}")
    return (
        X_automl,
        X_test_automl,
        X_train_automl,
        features_automl,
        le_automl,
        tmp_dataset_automl,
        y_automl,
        y_test_automl,
        y_train_automl,
    )


@app.cell
def _(mo):
    mo.md("""## 🚀 Entraînement AutoML""")
    return


@app.cell
def _(mo):
    mo.md("""
    ⏳ **En cours...** L'entraînement peut prendre 30-60 minutes.

    AutoGluon va :
    1. ✅ Tester 15+ algorithmes (LightGBM, CatBoost, XGBoost, RF, KNN, etc.)
    2. ✅ Optimiser hyperparamètres automatiquement
    3. ✅ Créer des ensembles (stacking) des meilleurs modèles
    4. ✅ Générer un leaderboard avec performances
    """)
    return


@app.cell
def _(CONFIG, X_test_automl, X_train_automl, pd, time, y_train_automl):
    from autogluon.tabular import TabularPredictor
    
    start_time_automl = time.time()
    
    # Créer le dataset pour AutoGluon (features + target)
    train_data_automl = pd.concat([X_train_automl, y_train_automl], axis=1)
    
    print(f"🚀 Démarrage AutoML ({CONFIG['time_limit']//60} min)...")
    print(f"   Métrique : {CONFIG['eval_metric']}")
    print(f"   Preset : {CONFIG['preset']}")
    
    # Entraîner AutoGluon
    predictor_automl = TabularPredictor(
        label='target',
        eval_metric=CONFIG['eval_metric'],
        problem_type='binary',
        path='./autogluon_models'
    ).fit(
        train_data=train_data_automl,
        time_limit=CONFIG['time_limit'],
        presets=CONFIG['preset'],
        verbosity=2
    )
    
    elapsed_time_automl = time.time() - start_time_automl
    
    print(f"\n✅ Entraînement terminé en {elapsed_time_automl/60:.1f} minutes")
    return (TabularPredictor, predictor_automl, start_time_automl, train_data_automl)


@app.cell
def _(mo):
    mo.md("""## 📊 Leaderboard des Modèles""")
    return


@app.cell
def _(X_test_automl, mo, predictor_automl):
    # Récupérer le leaderboard
    leaderboard_automl = predictor_automl.leaderboard(X_test_automl, silent=True)
    
    mo.md(f"""
    **Meilleur modèle** : `{predictor_automl.get_model_best()}`

    Comparaison de {len(leaderboard_automl)} modèles :
    """)
    
    # Afficher leaderboard avec colonnes principales
    display_cols = ['model', 'score_test', 'score_val', 'pred_time_test', 'fit_time']
    mo.ui.table(leaderboard_automl[display_cols].head(15))
    return (leaderboard_automl,)


@app.cell
def _(mo):
    mo.md("""## 🎯 Évaluation sur Test Set""")
    return


@app.cell
def _(
    X_test_automl,
    classification_report,
    confusion_matrix,
    f1_score,
    predictor_automl,
    precision_score,
    recall_score,
    roc_auc_score,
    y_test_automl,
):
    # Prédictions
    y_pred_automl = predictor_automl.predict(X_test_automl)
    y_proba_automl = predictor_automl.predict_proba(X_test_automl)
    
    # Métriques
    recall_automl = recall_score(y_test_automl, y_pred_automl)
    precision_automl = precision_score(y_test_automl, y_pred_automl)
    f1_automl = f1_score(y_test_automl, y_pred_automl)
    auc_automl = roc_auc_score(y_test_automl, y_proba_automl.iloc[:, 1])
    
    # Matrice de confusion
    cm_automl = confusion_matrix(y_test_automl, y_pred_automl)
    tn_automl, fp_automl, fn_automl, tp_automl = cm_automl.ravel()
    
    print("📈 PERFORMANCES AUTOML :")
    print("=" * 50)
    print(f"🎯 Recall (classe 1 - accidents)  : {recall_automl:.3f} ({tp_automl}/{tp_automl+fn_automl})")
    print(f"🎯 Precision (classe 1)           : {precision_automl:.3f}")
    print(f"🎯 F1-Score                        : {f1_automl:.3f}")
    print(f"🎯 AUC-ROC                         : {auc_automl:.3f}")
    print()
    print("🔍 Matrice de confusion :")
    print(f"   TN={tn_automl}, FP={fp_automl}")
    print(f"   FN={fn_automl}, TP={tp_automl}")
    print()
    print(classification_report(y_test_automl, y_pred_automl, target_names=['Pas Accident', 'Accident']))
    return (
        auc_automl,
        cm_automl,
        f1_automl,
        fn_automl,
        fp_automl,
        precision_automl,
        recall_automl,
        tn_automl,
        tp_automl,
        y_pred_automl,
        y_proba_automl,
    )


@app.cell
def _(cm_automl, fn_automl, fp_automl, mo, recall_automl, tn_automl, tp_automl):
    mo.callout(f"""
    **🏆 Résultats AutoML**

    **Détection accidents (classe 1)** :
    - ✅ **Recall : {recall_automl:.1%}** ({tp_automl}/{tp_automl+fn_automl} accidents détectés)
    - ⚠️ Faux négatifs : {fn_automl} accidents ratés
    - ⚠️ Faux positifs : {fp_automl} fausses alarmes

    **Matrice de confusion** :
    ```
                Prédit Non    Prédit Oui
    Réel Non      {tn_automl:5d}        {fp_automl:5d}
    Réel Oui      {fn_automl:5d}        {tp_automl:5d}
    ```
    """, kind="success" if recall_automl >= 0.87 else "warn")
    return


@app.cell
def _(mo):
    mo.md("""## 🌳 Feature Importance""")
    return


@app.cell
def _(mo, pd, predictor_automl):
    # Récupérer feature importance du meilleur modèle
    try:
        feature_importance_automl = predictor_automl.feature_importance(X_test_automl)
        
        # Créer DataFrame pour affichage
        fi_df_automl = pd.DataFrame({
            'feature': feature_importance_automl.index,
            'importance': feature_importance_automl.values,
            'importance_pct': (feature_importance_automl.values / feature_importance_automl.sum() * 100).round(1)
        }).sort_values('importance', ascending=False).head(15)
        
        mo.md("""
        **Top 15 features les plus importantes** :
        """)
        mo.ui.table(fi_df_automl)
        
    except Exception as e:
        mo.md(f"⚠️ Feature importance non disponible pour ce modèle : {e}")
    return


@app.cell
def _(mo):
    mo.md("""## 💾 Sauvegarde du Meilleur Modèle""")
    return


@app.cell
def _(joblib, predictor_automl):
    # Sauvegarder le meilleur modèle AutoML
    best_model_name_automl = predictor_automl.get_model_best()
    
    print(f"💾 Sauvegarde du meilleur modèle : {best_model_name_automl}")
    
    # AutoGluon sauvegarde automatiquement dans ./autogluon_models/
    # Pour une utilisation future, charger avec :
    # predictor = TabularPredictor.load('./autogluon_models/')
    
    # Exporter aussi un rapport
    report_automl = {
        'best_model': best_model_name_automl,
        'recall': recall_automl,
        'precision': precision_automl,
        'f1_score': f1_automl,
        'auc_roc': auc_automl,
        'confusion_matrix': {
            'tn': int(tn_automl), 'fp': int(fp_automl),
            'fn': int(fn_automl), 'tp': int(tp_automl)
        }
    }
    
    joblib.dump(report_automl, 'automl_report.pkl')
    print("✅ Rapport sauvegardé : automl_report.pkl")
    
    report_automl
    return (best_model_name_automl, report_automl)


@app.cell
def _(mo):
    mo.md("""
    ## 📝 Comparaison avec Optuna

    Pour comparer avec les résultats de `accident_fetch_data.py` (Optuna) :
    """)
    return


@app.cell
def _(auc_automl, f1_automl, mo, precision_automl, recall_automl):
    comparison_automl = {
        'Méthode': ['Optuna (CatBoost)', 'AutoML (AutoGluon)'],
        'Recall': [0.870, recall_automl],
        'Precision': [0.979, precision_automl],
        'F1-Score': [0.92, f1_automl],
        'AUC-ROC': [0.974, auc_automl]
    }
    
    import pandas as pd
    comparison_df_automl = pd.DataFrame(comparison_automl)
    
    mo.ui.table(comparison_df_automl)
    return (comparison_automl, comparison_df_automl)


@app.cell
def _(comparison_df_automl, mo, recall_automl):
    # Verdict
    optuna_recall = 0.870
    gain = (recall_automl - optuna_recall) * 100
    
    if recall_automl > optuna_recall:
        verdict_msg = f"🏆 **AutoML GAGNE** avec un gain de **+{gain:.1f}%** sur le recall !"
        verdict_kind = "success"
    elif recall_automl < optuna_recall:
        verdict_msg = f"⚠️ **Optuna reste meilleur** avec un avantage de **{-gain:.1f}%** sur le recall."
        verdict_kind = "warn"
    else:
        verdict_msg = "🤝 **Égalité parfaite** entre les deux méthodes."
        verdict_kind = "info"
    
    mo.callout(verdict_msg, kind=verdict_kind)
    return (gain, optuna_recall, verdict_kind, verdict_msg)


@app.cell
def _(mo):
    mo.md("""
    ## 🎯 Conclusion

    **Avantages AutoML** :
    - ✅ Zéro configuration hyperparamètres
    - ✅ Teste automatiquement 15+ algorithmes
    - ✅ Ensembles (stacking) automatiques
    - ✅ Rapport complet avec leaderboard

    **Avantages Optuna** :
    - ✅ Contrôle fin des hyperparamètres
    - ✅ Pruning pour gains de temps
    - ✅ Plus rapide (50 trials vs 30-60 min)
    - ✅ Modèles plus légers (un seul algorithme)

    **Recommandation** :
    - **Production** : Optuna (rapidité, légèreté, contrôle)
    - **Exploration** : AutoML (découvrir nouveaux algorithmes, benchmark complet)
    """)
    return


if __name__ == "__main__":
    app.run()
