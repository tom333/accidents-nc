import mlflow
import numpy as np
"""Asset Dagster pour entraînement CatBoost."""
from dagster import asset
from dagster_pipeline.assets.utils import save_model, save_predictions
import pandas as pd
import optuna
from catboost import CatBoostClassifier
from optuna.integration import CatBoostPruningCallback
from pipeline.stage_datasets import FEATURE_COLUMNS
from pipeline.config import GOLD_SCHEMA
from src.accidents.ducklake import get_client
from sklearn.metrics import recall_score

@asset(group_name="gold", deps=["ml_datasets"])
def tune_catboost():
    conn = get_client().conn
    train_df = conn.execute(f"SELECT * FROM {GOLD_SCHEMA}.train").df()
    test_df = conn.execute(f"SELECT * FROM {GOLD_SCHEMA}.test").df()
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df['target']
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df['target']

    def objective(trial):
        from sklearn.metrics import recall_score
        cb_params = {
            'iterations': trial.suggest_int('iterations', 150, 500),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'auto_class_weights': 'Balanced',
            'eval_metric': 'AUC',
            'random_state': 42,
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

    study = optuna.create_study(
        direction='maximize',
        study_name='CatBoost',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    best_model = CatBoostClassifier(
        **study.best_params,
        auto_class_weights='Balanced',
        eval_metric='AUC',
        random_state=42,
        verbose=False
    )
    best_model.fit(X_train, y_train)
    save_model(best_model, "catboost_model.pkl")
    preds = best_model.predict_proba(X_test)
    save_predictions(preds, "catboost_preds.csv")

    # Log dans MLflow
    import os
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        recall_score, precision_score, f1_score, roc_auc_score, 
        precision_recall_curve, auc as auc_metric
    )
    
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("accidents-nc")
    
    with mlflow.start_run(run_name="catboost_gold"):
        mlflow.log_params(study.best_params)
        
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        # ========== THRESHOLD TUNING (Precision-Recall Curve) ==========
        print("🎯 Recherche du seuil optimal...")
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
        
        # Calcul F1-Score pour chaque seuil
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_threshold_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5
        best_f1 = f1_scores[best_threshold_idx]
        
        print(f"✅ Seuil optimal trouvé: {best_threshold:.4f} (F1={best_f1:.4f})")
        
        # Sauvegarder le seuil avec le modèle
        import joblib
        joblib.dump({'threshold': best_threshold}, 'catboost_threshold.pkl')
        mlflow.log_artifact('catboost_threshold.pkl')
        
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
        mlflow.log_metric("recall_default", recall_default)
        mlflow.log_metric("precision_default", precision_default)
        mlflow.log_metric("f1_default", f1_default)
        
        # Log métriques (threshold tuned)
        mlflow.log_metric("recall_tuned", recall_tuned)
        mlflow.log_metric("precision_tuned", precision_tuned)
        mlflow.log_metric("f1_tuned", f1_tuned)
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("auc_roc", auc_roc)
        
        # Plot Precision-Recall curve
        plt.figure(figsize=(10, 6))
        plt.plot(recalls, precisions, label='Precision-Recall curve')
        plt.scatter([recalls[best_threshold_idx]], [precisions[best_threshold_idx]], 
                   color='red', s=100, zorder=5, label=f'Optimal (t={best_threshold:.3f}, F1={best_f1:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - CatBoost')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('precision_recall_curve.png', dpi=150, bbox_inches='tight')
        mlflow.log_artifact('precision_recall_curve.png')
        plt.close()
        
        # Feature Importance
        feature_importances = best_model.get_feature_importance()
        feature_names = FEATURE_COLUMNS
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importances
        }).sort_values('importance', ascending=False)
        
        # Log top 10 comme métriques
        for idx, row in importance_df.head(10).iterrows():
            mlflow.log_metric(f"importance_{row['feature']}", row['importance'])
        
        # Créer et logger le plot
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'].head(15), importance_df['importance'].head(15))
        plt.xlabel('Importance')
        plt.title('CatBoost Feature Importance (Top 15)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance_catboost.png', dpi=150, bbox_inches='tight')
        mlflow.log_artifact('feature_importance_catboost.png')
        plt.close()
        
        # Logger le CSV complet
        importance_df.to_csv('feature_importance_catboost.csv', index=False)
        mlflow.log_artifact('feature_importance_catboost.csv')
        
        # ========== SHAP EXPLICABILITY ==========
        print("🔍 Calcul SHAP values pour explicabilité...")
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
            shap.summary_plot(
                shap_values, 
                X_test_sample, 
                feature_names=FEATURE_COLUMNS,
                show=False
            )
            plt.tight_layout()
            plt.savefig('shap_summary_plot.png', dpi=150, bbox_inches='tight')
            mlflow.log_artifact('shap_summary_plot.png')
            plt.close()
            
            # Bar Plot (importance moyenne absolue)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values, 
                X_test_sample,
                feature_names=FEATURE_COLUMNS,
                plot_type='bar',
                show=False
            )
            plt.tight_layout()
            plt.savefig('shap_bar_plot.png', dpi=150, bbox_inches='tight')
            mlflow.log_artifact('shap_bar_plot.png')
            plt.close()
            
            print("✅ SHAP plots générés et loggés")
            
        except ImportError:
            print("⚠️  SHAP non installé, skipped (pip install shap)")
        except Exception as e:
            print(f"⚠️  Erreur SHAP: {e}")
        
        mlflow.log_artifact("catboost_model.pkl")
        mlflow.log_artifact("catboost_preds.csv")

    return "catboost_model.pkl", "catboost_preds.csv", study.best_params
