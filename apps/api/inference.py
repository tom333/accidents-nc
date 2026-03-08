import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.utils.temporal import compute_temporal_features

from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)

# Paths to artifacts - using absolute path to project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
KMEANS_PATH = BASE_DIR / "kmeans_geo.pkl"
ATM_ENCODER_PATH = BASE_DIR / "atm_encoder.pkl"
CATBOOST_PATH = BASE_DIR / "catboost_model.pkl"
XGBOOST_PATH = BASE_DIR / "xgboost_model.pkl"
MLP_PATH = BASE_DIR / "mlp_model.pkl"
# On utilise maintenant le seuil optimal du blend
BLEND_METRICS_PATH = BASE_DIR / "blend_metrics.json"
FEATURES_PATH = BASE_DIR / "features.pkl"


class InferenceService:
    def __init__(self):
        self.models = {}
        self.kmeans = None
        self.atm_encoder = None
        self.features = []
        self.threshold = 0.5
        self.is_ready = False
        # Do not load during init to avoid issues with missing files at startup
        # if the job is still running.

    def load_artifacts(self):
        """Charger les modèles et encodeurs depuis le disque."""
        print(f"🔍 Tentative de chargement des artefacts depuis {BASE_DIR}...")
        try:
            if KMEANS_PATH.exists():
                print(f"✅ Trouvé: {KMEANS_PATH.name}")
                self.kmeans = joblib.load(KMEANS_PATH)
            else:
                print(f"❌ Manquant: {KMEANS_PATH.name}")

            if ATM_ENCODER_PATH.exists():
                self.atm_encoder = joblib.load(ATM_ENCODER_PATH)

            if FEATURES_PATH.exists():
                self.features = joblib.load(FEATURES_PATH)

            if BLEND_METRICS_PATH.exists():
                with open(BLEND_METRICS_PATH) as f:
                    meta = json.load(f)
                self.threshold = meta.get("threshold", 0.5)
                print(f"✅ Seuil du blend chargé: {self.threshold}")

            # Modèles
            for name, path in [
                ("catboost", CATBOOST_PATH),
                ("xgboost", XGBOOST_PATH),
                ("mlp", MLP_PATH),
            ]:
                if path.exists():
                    print(f"✅ Chargé: {name}")
                    self.models[name] = joblib.load(path)
                else:
                    print(f"❌ Manquant: {name} ({path.name})")

            self.is_ready = len(self.models) > 0 and self.kmeans is not None
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

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Pipeline complet d'inférence."""
        if not self.is_ready:
            raise RuntimeError("Le service d'inférence n'est pas prêt (artefacts manquants)")

        # 1. Feature Engineering
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

        # ATM Encoding
        try:
            df["atm"] = self.atm_encoder.transform(df["atm"].astype(str))
        except Exception:
            # Fallback si valeur inconnue (souvent 0 pour 'Normale')
            df["atm"] = 0

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

        # 1. Feature Engineering
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

        try:
            df["atm"] = self.atm_encoder.transform(df["atm"].astype(str))
        except Exception:
            df["atm"] = 0

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
