import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.utils.temporal import compute_temporal_features

from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)

# Paths to artifacts
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_BASE_PATH = Path(os.getenv("MODEL_BASE_PATH", "/app/models"))

# Runtime-mounted artifacts (preferred)
KMEANS_PATH = MODEL_BASE_PATH / "kmeans_geo.pkl"
CATEGORICAL_ENCODERS_PATH = MODEL_BASE_PATH / "categorical_encoders.pkl"
ROUTES_GRID_PATH = MODEL_BASE_PATH / "routes_grid.pkl"
BLEND_METRICS_PATH = MODEL_BASE_PATH / "blend_metrics.json"
FEATURES_PATH = MODEL_BASE_PATH / "features.pkl"

CATBOOST_MLFLOW_DIR = MODEL_BASE_PATH / "catboost"
XGBOOST_MLFLOW_DIR = MODEL_BASE_PATH / "xgboost"
MLP_MLFLOW_DIR = MODEL_BASE_PATH / "mlp"

# Backward-compatible local fallbacks
CATBOOST_PKL_FALLBACK = BASE_DIR / "catboost_model.pkl"
XGBOOST_PKL_FALLBACK = BASE_DIR / "xgboost_model.pkl"
MLP_PKL_FALLBACK = BASE_DIR / "mlp_model.pkl"
KMEANS_PKL_FALLBACK = BASE_DIR / "kmeans_geo.pkl"
CATEGORICAL_ENCODERS_PKL_FALLBACK = BASE_DIR / "categorical_encoders.pkl"
ROUTES_GRID_PKL_FALLBACK = BASE_DIR / "routes_grid.pkl"
FEATURES_PKL_FALLBACK = BASE_DIR / "features.pkl"
BLEND_METRICS_FALLBACK = BASE_DIR / "blend_metrics.json"


class InferenceService:
    def __init__(self):
        self.models = {}
        self.kmeans = None
        self.encoders = {}
        self.routes_grid = None
        self.routes_tree = None
        self.features = []
        self.threshold = 0.5
        self.is_ready = False

    def load_artifacts(self):
        """Charger les modèles et encodeurs depuis le disque."""
        print(
            f"🔍 Tentative de chargement des artefacts depuis volume={MODEL_BASE_PATH} "
            f"(fallback={BASE_DIR})..."
        )
        try:
            kmeans_path = KMEANS_PATH if KMEANS_PATH.exists() else KMEANS_PKL_FALLBACK
            if kmeans_path.exists():
                print(f"✅ Trouvé: {kmeans_path}")
                self.kmeans = joblib.load(kmeans_path)
            else:
                print(f"❌ Manquant: {KMEANS_PATH.name}")

            encoders_path = (
                CATEGORICAL_ENCODERS_PATH
                if CATEGORICAL_ENCODERS_PATH.exists()
                else CATEGORICAL_ENCODERS_PKL_FALLBACK
            )
            if encoders_path.exists():
                print(f"✅ Trouvé: {encoders_path}")
                self.encoders = joblib.load(encoders_path)
            else:
                print(f"❌ Manquant: {CATEGORICAL_ENCODERS_PATH.name}")

            routes_path = (
                ROUTES_GRID_PATH if ROUTES_GRID_PATH.exists() else ROUTES_GRID_PKL_FALLBACK
            )
            if routes_path.exists():
                print(f"✅ Trouvé: {routes_path}")
                self.routes_grid = joblib.load(routes_path)
                # Build tree for fast nearest neighbor search
                grid_coords = self.routes_grid[["latitude", "longitude"]].to_numpy()
                self.routes_tree = cKDTree(grid_coords)
            else:
                print(f"❌ Manquant: {ROUTES_GRID_PATH.name}")

            features_path = FEATURES_PATH if FEATURES_PATH.exists() else FEATURES_PKL_FALLBACK
            if features_path.exists():
                self.features = joblib.load(features_path)

            blend_metrics_path = (
                BLEND_METRICS_PATH if BLEND_METRICS_PATH.exists() else BLEND_METRICS_FALLBACK
            )
            if blend_metrics_path.exists():
                with open(blend_metrics_path) as f:
                    meta = json.load(f)
                self.threshold = meta.get("threshold", 0.5)
                print(f"✅ Seuil du blend chargé: {self.threshold}")

            # Modèles MLflow (prioritaire) + fallback pkl
            try:
                import mlflow.catboost
                import mlflow.pytorch
                import mlflow.xgboost

                if (CATBOOST_MLFLOW_DIR / "MLmodel").exists():
                    self.models["catboost"] = mlflow.catboost.load_model(str(CATBOOST_MLFLOW_DIR))
                    print(f"✅ Chargé catboost depuis {CATBOOST_MLFLOW_DIR}")

                if (XGBOOST_MLFLOW_DIR / "MLmodel").exists():
                    self.models["xgboost"] = mlflow.xgboost.load_model(str(XGBOOST_MLFLOW_DIR))
                    print(f"✅ Chargé xgboost depuis {XGBOOST_MLFLOW_DIR}")

                if (MLP_MLFLOW_DIR / "MLmodel").exists():
                    self.models["mlp"] = mlflow.pytorch.load_model(str(MLP_MLFLOW_DIR))
                    print(f"✅ Chargé mlp depuis {MLP_MLFLOW_DIR}")
            except Exception as e:
                print(f"⚠️ Chargement MLflow partiel/échoué: {e}")

            for name, path in [
                ("catboost", CATBOOST_PKL_FALLBACK),
                ("xgboost", XGBOOST_PKL_FALLBACK),
                ("mlp", MLP_PKL_FALLBACK),
            ]:
                if name not in self.models and path.exists():
                    print(f"✅ Fallback local chargé: {name} ({path})")
                    self.models[name] = joblib.load(path)
                elif name not in self.models:
                    print(f"❌ Manquant: {name} ({path.name})")

            expected_models = {"catboost", "xgboost", "mlp"}
            self.is_ready = (
                expected_models.issubset(set(self.models.keys()))
                and self.kmeans is not None
                and len(self.encoders) > 0
                and self.routes_tree is not None
                and len(self.features) > 0
            )
            print(f"🚀 Service prêt: {self.is_ready}")
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"❌ Erreur chargement: {e}")
            self.is_ready = False

    def _get_risk_level(self, proba: float) -> str:
        if proba < self.threshold * 0.5:
            return "Faible"
        if proba < self.threshold:
            return "Modéré"
        return "Élevé"

    def _attach_road_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trouve la route la plus proche et copie ses features routières."""
        coords = df[["latitude", "longitude"]].to_numpy()
        _, idx = self.routes_tree.query(coords, k=1)
        nearest = self.routes_grid.iloc[idx].reset_index(drop=True)

        df["road_type"] = nearest["road_type"].to_numpy()
        df["speed_limit"] = nearest["speed_limit"].to_numpy()
        df["lanes"] = nearest["lanes"].to_numpy()
        df["lit"] = nearest["lit"].to_numpy()
        df["surface"] = nearest["surface"].to_numpy()
        df["oneway"] = nearest["oneway"].to_numpy()
        return df

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Pipeline complet d'inférence."""
        if not self.is_ready:
            raise RuntimeError("Le service d'inférence n'est pas prêt (artefacts manquants)")

        # 1. Feature Engineering Base
        ts = request.timestamp
        data = {
            "latitude": [request.latitude],
            "longitude": [request.longitude],
            "atm": [request.atm],
            "datetime": [ts],
            "hour": [ts.hour],
            "dayofweek": [ts.weekday()],
            "month": [ts.month],
        }
        df = pd.DataFrame(data)

        # Temporel
        compute_temporal_features(df)

        # Spatial (Geo-Clustering)
        coords = df[["latitude", "longitude"]].values
        df["geo_cluster"] = self.kmeans.predict(coords)

        # Attacher features routières spatiales
        df = self._attach_road_features(df)

        # Encodage de toutes les variables catégorielles
        for feature, encoder in self.encoders.items():
            if feature in df.columns:
                try:
                    df[feature] = encoder.transform(df[feature].astype(str))
                except Exception:
                    # Fallback sur la première classe (index 0) si inconnue
                    df[feature] = 0

        # Garder uniquement les features nécessaires dans le bon ordre
        X = df[self.features]

        # 2. Inférence Ensemble (Blending)
        probas = []
        weights = {"catboost": 0.4, "xgboost": 0.4, "mlp": 0.2}
        total_weight = 0

        for name, model in self.models.items():
            w = weights.get(name, 0)
            if name == "mlp":
                # Le wrapper MLP s'attend à un tableau numpy
                p = model.predict_proba(X.values)[:, 1][0]
            else:
                p = model.predict_proba(X)[:, 1][0]

            probas.append(p * w)
            total_weight += w

        final_proba = sum(probas) / total_weight if total_weight > 0 else 0

        return PredictionResponse(
            probability=float(final_proba),
            risk_level=self._get_risk_level(final_proba),
            threshold=float(self.threshold),
            is_dangerous=bool(final_proba >= self.threshold),
            model_version="1.0.0-ensemble",
        )

    def predict_batch(self, request: BatchPredictionRequest) -> BatchPredictionResponse:
        """Pipeline complet d'inférence pour un lot de points."""
        if not self.is_ready:
            raise RuntimeError("Le service n'est pas prêt")

        # 1. Feature Engineering Base
        ts = request.timestamp
        lats = [loc["latitude"] for loc in request.locations]
        lons = [loc["longitude"] for loc in request.locations]

        df = pd.DataFrame(
            {
                "latitude": lats,
                "longitude": lons,
                "atm": [request.atm] * len(lats),
                "datetime": [ts] * len(lats),
                "hour": [ts.hour] * len(lats),
                "dayofweek": [ts.weekday()] * len(lats),
                "month": [ts.month] * len(lats),
            }
        )

        compute_temporal_features(df)
        df["geo_cluster"] = self.kmeans.predict(df[["latitude", "longitude"]].values)

        # Attacher features routières
        df = self._attach_road_features(df)

        # Encodage
        for feature, encoder in self.encoders.items():
            if feature in df.columns:
                try:
                    df[feature] = encoder.transform(df[feature].astype(str))
                except Exception:
                    df[feature] = 0

        X = df[self.features]

        # 2. Inférence Ensemble
        all_probas = []
        weights = {"catboost": 0.4, "xgboost": 0.4, "mlp": 0.2}
        total_weight = 0

        for name, model in self.models.items():
            w = weights.get(name, 0)
            if name == "mlp":
                p = model.predict_proba(X.values)[:, 1]
            else:
                p = model.predict_proba(X)[:, 1]

            all_probas.append(p * w)
            total_weight += w

        final_probas = sum(all_probas) / total_weight if total_weight > 0 else np.zeros(len(df))

        return BatchPredictionResponse(
            predictions=final_probas.tolist(),
            threshold=float(self.threshold),
            model_version="1.0.0-ensemble",
        )


# Singleton
inference_service = InferenceService()
