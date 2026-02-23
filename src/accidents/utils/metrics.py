"""Utilitaires métriques ML."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict:
    """
    Calcule les métriques de classification.
    
    Args:
        y_true: Labels réels
        y_pred: Prédictions binaires
        y_proba: Probabilités prédites (optionnel, pour AUC)
        
    Returns:
        Dictionnaire de métriques
    """
    metrics = {}
    
    # Matrice de confusion
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negative'] = int(tn)
    metrics['false_positive'] = int(fp)
    metrics['false_negative'] = int(fn)
    metrics['true_positive'] = int(tp)
    
    # Métriques dérivées
    metrics['accuracy'] = float((tp + tn) / (tp + tn + fp + fn))
    metrics['precision'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    metrics['recall'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics['f1_score'] = float(
        2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        if (metrics['precision'] + metrics['recall']) > 0 else 0.0
    )
    
    # AUC-ROC si probabilités disponibles
    if y_proba is not None:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))
        metrics['average_precision'] = float(average_precision_score(y_true, y_proba))
    
    return metrics


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str] | None = None) -> None:
    """
    Affiche le rapport de classification sklearn.
    
    Args:
        y_true: Labels réels
        y_pred: Prédictions
        target_names: Noms des classes (optionnel)
    """
    print(classification_report(y_true, y_pred, target_names=target_names))


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray, metric: str = 'f1') -> tuple[float, float]:
    """
    Trouve le seuil optimal pour maximiser une métrique.
    
    Args:
        y_true: Labels réels
        y_proba: Probabilités prédites
        metric: 'f1', 'precision' ou 'recall'
        
    Returns:
        Tuple (seuil_optimal, score_métrique)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    if metric == 'f1':
        scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    elif metric == 'precision':
        scores = precision
    elif metric == 'recall':
        scores = recall
    else:
        raise ValueError("metric doit être 'f1', 'precision' ou 'recall'")
    
    # Thresholds a une taille N-1 par rapport aux scores
    best_idx = np.argmax(scores[:-1])
    optimal_threshold = float(thresholds[best_idx])
    best_score = float(scores[best_idx])
    
    return optimal_threshold, best_score


def compute_spatial_coverage(predictions_gdf, buffer_km: float = 1.0) -> float:
    """
    Calcule la couverture spatiale des prédictions positives.
    
    Args:
        predictions_gdf: GeoDataFrame avec géométrie et colonne 'prediction'
        buffer_km: Buffer en km pour calculer l'aire couverte
        
    Returns:
        Aire couverte en km²
    """
    positive_preds = predictions_gdf[predictions_gdf['prediction'] == 1]
    
    if len(positive_preds) == 0:
        return 0.0
    
    # Reprojeter en mètres
    gdf_projected = positive_preds.to_crs(epsg=3857)
    
    # Buffer et union
    buffered = gdf_projected.geometry.buffer(buffer_km * 1000)
    coverage = buffered.unary_union.area / 1e6  # m² → km²
    
    return float(coverage)
