"""Asset Dagster pour l'export du résultat du blending.

Puisque le blending est une opération qui produit des prédictions
(`blend_preds.csv`) et non un modèle entraîné unique, cet asset se contente
de l'exposer comme artefact de sortie pour la suite du pipeline.
"""
from dagster import asset


@asset(group_name="gold", deps=["blend_model"])
def export_blend():
    # L'asset `blend_model` produit `blend_preds.csv` dans le répertoire de
    # travail ; on renvoie simplement ce fichier comme artefact d'export.
    # Note : cet asset est volontairement léger — il sert de point d'intégration
    # dans la DAG pour exposer/nommer les prédictions produites par `blend_model`.
    # Si on veut versionner / uploader / packager les prédictions (ex. MLflow,
    # S3, ou renommage avec timestamp), implémenter cette logique ici.
    return "blend_preds.csv"
