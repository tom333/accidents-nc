"""Asset Dagster pour la génération de rapports de qualité Evidently AI."""

import json
from pathlib import Path

import pandas as pd
from dagster import AssetExecutionContext, asset
from evidently import Report
from evidently.presets import ClassificationPreset

from src.assets.gold.schema import GOLD_SCHEMA
from src.ducklake import get_client


@asset(
    group_name="report",
    required_resource_keys={"mlflow"},
    deps=["blend_model"],  # Dépendance sur le blend pour avoir les prédictions
)
def evidently_report(context: AssetExecutionContext):
    """Génère un rapport de qualité enrichi avec Evidently AI (Classification, Drift, Summary)."""

    # ─── Chargement des données ───
    conn = get_client().conn
    test_df = conn.execute(f"SELECT * FROM {GOLD_SCHEMA}.test").df()
    train_df = conn.execute(f"SELECT * FROM {GOLD_SCHEMA}.train").df()

    # Charger les probabilités du blend
    preds_df = pd.read_csv("blend_preds.csv")
    if "prediction" in preds_df.columns:
        proba_blend = preds_df["prediction"].values.astype(float)
    else:
        proba_blend = preds_df.iloc[:, 0].values.astype(float)

    # Charger le seuil optimal
    threshold = 0.5
    if Path("blend_metrics.json").exists():
        try:
            with open("blend_metrics.json") as f:
                metrics = json.load(f)
            threshold = float(metrics.get("threshold", 0.5))
        except Exception as e:
            context.log.warning(f"Erreur lecture blend_metrics.json: {e}, utilisation 0.5")

    # Créer les prédictions binaires
    y_pred = (proba_blend >= threshold).astype(int)

    # Préparer le DataFrame de test pour Evidently
    report_df = test_df.copy()
    report_df["prediction"] = y_pred
    report_df["proba"] = proba_blend

    # Préparer le DataFrame de train (référence)
    # Evidently exige que les colonnes de prédiction soient présentes
    # dans les deux datasets si elles sont définies dans DataDefinition.
    train_df = train_df.copy()
    train_df["prediction"] = train_df["target"]
    train_df["proba"] = train_df["target"].astype(float)

    # ─── Génération du rapport ───
    context.log.info("📊 Génération du rapport Evidently enrichi avec explications...")

    from evidently import BinaryClassification, DataDefinition, Dataset
    from evidently.core.container import MetricContainer
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    class Comment(MetricContainer):
        """Classe wrapper pour ajouter du texte Markdown dans un rapport Evidently v2."""

        comment_text: str

        def __init__(self, text: str):
            self.comment_text = text
            super().__init__(include_tests=False)

        def generate_metrics(self, context):
            return []

        def render(self, context, child_widgets=None):
            from evidently.legacy.metrics.data_drift.text_metric import Comment as LegacyComment

            _, render = context.get_legacy_metric(
                LegacyComment(text=self.comment_text), input_data_generator=None, task_name=None
            )
            return render

    # Configuration de la tâche de classification
    data_definition = DataDefinition(
        classification=[
            BinaryClassification(
                target="target",
                prediction_labels="prediction",
                prediction_probas="proba",
                pos_label=1,
            )
        ]
    )

    # Création des Datasets
    current_dataset = Dataset.from_pandas(report_df, data_definition=data_definition)
    reference_dataset = Dataset.from_pandas(train_df, data_definition=data_definition)

    # Rapport avec commentaires explicatifs via notre classe personnalisée
    classification_report = Report(
        metrics=[
            Comment(
                text="""# 📋 Rapport de Performance du Modèle de Blending
Ce rapport évalue la qualité du modèle combiné (Blended Model) sur le jeu de test, en le comparant aux données d'entraînement.
Il contient trois sections principales :
1. **Performance de Classification** : Analyse de la précision, du rappel et des courbes ROC/PR.
2. **Résumé des Données** : Statistiques descriptives sur les variables prédictives.
3. **Analyse du Drift** : Détection de changements de distribution entre train et test."""
            ),
            Comment(text="## 🎯 Section 1 : Performance de Classification"),
            ClassificationPreset(),
            Comment(
                text="""**Note sur la Classification** :
Ce dashboard présente les métriques clés pour évaluer la capacité du modèle à identifier les accidents :
- **Précision (Precision)** : Proportion d'accidents réels parmi ceux prédits par le modèle. Une valeur de 0.8 signifie que 80% des alertes sont vraies. C'est crucial pour éviter les "fausses alertes".
- **Rappel (Recall)** : Proportion d'accidents réels que le modèle a réussi à capturer. Un rappel de 0.9 signifie que le modèle détecte 90% des accidents. C'est vital pour la sécurité (ne rien rater).
- **F1-Score** : Moyenne équilibrée entre Précision et Rappel. Utilité : donne une vision globale de la performance, surtout si les classes sont déséquilibrées.
- **ROC-AUC** : Capacité du modèle à distinguer les classes. Une valeur de 0.5 est un hasard total, **0.8+ est considéré comme très bon**, et 0.9+ est excellent.

*Qu'est-ce qu'une "bonne" valeur ?* Cela dépend du métier. Ici, on cherche un compromis : un rappel élevé pour la sécurité, sans sacrifier totalement la précision pour garder la crédibilité des alertes."""
            ),
            Comment(text="## 📊 Section 2 : Résumé des Caractéristiques (Feature Stats)"),
            DataSummaryPreset(),
            Comment(text="## 📉 Section 3 : Analyse du Drift (Data Drift)"),
            DataDriftPreset(),
            Comment(
                text="""**Note sur le Drift** :
Le Drift indique si vos données de test sont statistiquement différentes de vos données d'entraînement.
Un drift élevé sur des features importantes peut expliquer une baisse de performance en production."""
            ),
        ]
    )

    # Exécution du rapport
    snapshot = classification_report.run(
        current_data=current_dataset, reference_data=reference_dataset
    )

    report_filename = "evidently_report.html"
    snapshot.save_html(report_filename)

    # ─── Logging MLflow ───
    mlflow = context.resources.mlflow
    mlflow.log_artifact(report_filename)

    context.log.info(f"✅ Rapport Evidently verbeux généré et loggé : {report_filename}")

    return {"report_path": report_filename}
