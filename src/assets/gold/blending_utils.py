"""Utilitaires pour le blending d'ensemble ML."""

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin

from src.utils.models import load_model


def predict_blend(X, model_paths, weights=(0.4, 0.4, 0.2)):
    """Calcule la prédiction d'ensemble pondérée sur X."""
    catboost = load_model(model_paths[0])
    xgboost = load_model(model_paths[1])
    mlp = load_model(model_paths[2])

    # CatBoost doit recevoir un DataFrame (avec noms de colonnes) quand le modèle
    # a été entraîné avec des features catégorielles déclarées via cat_features.
    X_df = X
    X_np = X.values if hasattr(X, "values") else np.asarray(X)

    probas_cat = catboost.predict_proba(X_df)[:, 1]
    probas_xgb = xgboost.predict_proba(X_df)[:, 1]
    probas_mlp = mlp.predict_proba(X_np)[:, 1]

    w_cat, w_xgb, w_mlp = weights
    proba_blend = (probas_cat * w_cat + probas_xgb * w_xgb + probas_mlp * w_mlp) / (
        w_cat + w_xgb + w_mlp
    )
    return proba_blend


class BlendingEnsembleWrapper(BaseEstimator, ClassifierMixin):
    """
    Une classe "Wrapper" compatible Scikit-Learn qui encapsule
    CatBoost, XGBoost et le MLP PyTorch avec la logique de vote pondéré.
    """

    def __init__(self, cat_model, xgb_model, mlp_model, mlp_weights, threshold=0.73):
        self.cat_model = cat_model
        self.xgb_model = xgb_model
        self.mlp_model = mlp_model
        self.mlp_weights = mlp_weights  # ex: [0.4, 0.4, 0.2]
        self.threshold = threshold

        # Pour le MLP
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # On définit les classes pour Scikit-Learn / Deepchecks
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        """Prend un DataFrame X et retourne les probabilités combinées."""
        # 1. CatBoost & XGBoost
        # On suppose que X contient déjà les bonnes colonnes (y compris geo_cluster)
        p_cat = self.cat_model.predict_proba(X)[:, 1]
        p_xgb = self.xgb_model.predict_proba(X)[:, 1]

        # 2. Préparation des données pour le MLP
        # On sépare le geo_cluster (cat) des autres numériques
        X_cat = X["geo_cluster"].values
        num_cols = [c for c in X.columns if c != "geo_cluster"]
        X_num = X[num_cols].values

        X_cat_t = torch.LongTensor(X_cat).to(self.device)
        X_num_t = torch.FloatTensor(X_num).to(self.device)

        # 3. Prédiction MLP
        self.mlp_model.eval()
        with torch.no_grad():
            p_mlp = self.mlp_model(X_cat_t, X_num_t).cpu().numpy()

        # 4. Le Blending (Vote Pondéré)
        w_cat, w_xgb, w_mlp = self.mlp_weights
        p_ensemble = (p_cat * w_cat) + (p_xgb * w_xgb) + (p_mlp * w_mlp)
        # Division par la somme des poids (si la somme n'est pas 1)
        p_ensemble = p_ensemble / sum(self.mlp_weights)

        # Deepchecks et Sklearn attendent un tableau 2D [proba_0, proba_1]
        return np.column_stack([1 - p_ensemble, p_ensemble])

    def predict(self, X):
        """Retourne 0 ou 1 basé sur le seuil optimal calculé."""
        probas = self.predict_proba(X)[:, 1]
        return (probas >= self.threshold).astype(int)
