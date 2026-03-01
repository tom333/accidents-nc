import pandas as pd
from dagster import asset, Failure, AssetExecutionContext, MetadataValue
from deepchecks.tabular import Dataset
from deepchecks.core import Suite
from deepchecks.tabular.checks import TrainTestPerformance, FeatureLabelCorrelation
import mlflow
import os
from pipeline.stage_datasets import FEATURE_COLUMNS
from pipeline.config import GOLD_SCHEMA
from src.accidents.ducklake import get_client

@asset(
    required_resource_keys={"mlflow"}, # Si vous utilisez la ressource MLflow de Dagster
    description="Vérifie l'intégrité du modèle avec Deepchecks avant la mise en production.",
    deps=["export_blend", "eval_blend"]
)
def validate_and_register_model(
    context: AssetExecutionContext, 
    trained_ensemble_model # Votre modèle final issu de l'asset précédent
):
    context.log.info("🛡️ Démarrage de la Quality Gate Deepchecks...")

    # 1. Envelopper les données pour Deepchecks
    # Assurez-vous que X_train_clean contient bien 'geo_cluster' et pas de GPS bruts
    conn = get_client().conn
    test_df = conn.execute(f"SELECT * FROM {GOLD_SCHEMA}.test").df()
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df['target'].values
    # 2. Création de la Suite de Tests avec Conditions Stricts (Pass/Fail)
    validation_suite = Suite("Accidents Quality Gate")
    
    # Check A : Y a-t-il une fuite de données flagrante ?
    # Le PPS (Predictive Power Score) mesure la capacité d'une seule feature à prédire la target.
    # On fait échouer si une feature a un score > 0.70
    leakage_check = FeatureLabelCorrelation().add_condition_feature_pps_less_than(0.70)
    
    # Check B : Le modèle est-il suffisamment performant et robuste ?
    # On exige une AUC minimum de 0.85 sur le Test Set
    performance_check = TrainTestPerformance().add_condition_test_performance_greater_than(0.85)

    validation_suite.add(leakage_check)
    validation_suite.add(performance_check)

    # 3. Exécution de la suite
    result = validation_suite.run(train_dataset=ds_train, test_dataset=ds_test, model=trained_ensemble_model)
    
    # 4. Sauvegarde du rapport pour MLflow
    report_path = "deepchecks_validation_report.html"
    result.save_as_html(report_path)
    
    # Log du rapport HTML dans l'UI MLflow (pour que le Data Scientist puisse déboguer)
    mlflow.log_artifact(report_path)
    os.remove(report_path) # Nettoyage local

    # 5. Le "Videur" : Analyse des résultats
    if not result.passed():
        # Si un test échoue, on crashe le pipeline Dagster proprement
        failed_checks = [check.get_header() for check in result.get_not_passed_checks()]
        
        error_msg = f"❌ Le modèle n'a pas passé les tests de qualité : {', '.join(failed_checks)}"
        context.log.error(error_msg)
        
        raise Failure(
            description=error_msg,
            metadata={
                "action": "Consultez l'artefact HTML dans MLflow pour voir les détails de l'échec.",
            }
        )

    # 6. Succès ! Enregistrement dans le Model Registry
    context.log.info("✅ Tests Deepchecks réussis. Enregistrement du modèle en cours...")
    
    # On enregistre le modèle officiellement
    mlflow.sklearn.log_model(
        sk_model=trained_ensemble_model,
        artifact_path="final_ensemble_model",
        registered_model_name="Accidents_NC_Predictor" # Crée une version V1, V2, etc.
    )

    # Métadonnées pour l'UI Dagster
    context.add_output_metadata({
        "status": "Validated and Registered",
        "deepchecks_passed": True
    })

    return True