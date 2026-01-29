"""
API de prédiction de risque d'accidents - Nouvelle-Calédonie
Charge automatiquement le modèle en production depuis MLflow
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os

app = FastAPI(
    title="Accident Prediction API - Nouvelle-Calédonie",
    description="Prédiction du risque d'accidents routiers",
    version="2.0.0"
)

# Configuration
MLFLOW_URI = os.getenv("MLFLOW_URI", "http://mlflow:5000")
MODEL_NAME = "accident_prediction_nc"

# Cache du modèle
model_cache = {
    "model": None,
    "version": None,
    "features": None,
    "encoder": None,
    "loaded_at": None
}


class PredictionRequest(BaseModel):
    latitude: float = Field(..., ge=-23.0, le=-19.5, description="Latitude en Nouvelle-Calédonie")
    longitude: float = Field(..., ge=163.5, le=168.0, description="Longitude en Nouvelle-Calédonie")
    datetime: str = Field(..., description="Date/heure au format ISO 8601", example="2026-01-24T14:30:00")
    atm: int = Field(..., ge=1, le=9, description="Conditions atmosphériques (1-9)")


class PredictionResponse(BaseModel):
    risk_score: float = Field(..., description="Score de risque (0-1)")
    risk_level: str = Field(..., description="Niveau de risque: LOW, MEDIUM, HIGH, CRITICAL")
    confidence: float = Field(..., description="Confiance de la prédiction")
    model_version: str = Field(..., description="Version du modèle utilisée")
    features_used: Dict = Field(..., description="Features extraites")


def load_production_model():
    """Charger le modèle en production depuis MLflow"""
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()
        
        # Récupérer la version en production
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not versions:
            raise ValueError("Aucun modèle en production")
        
        latest_version = versions[0]
        
        # Charger uniquement si nouvelle version
        if model_cache["version"] != latest_version.version:
            print(f"🔄 Chargement du modèle v{latest_version.version}...")
            
            model_uri = f"models:/{MODEL_NAME}/Production"
            model = mlflow.sklearn.load_model(model_uri)
            
            # Télécharger les artefacts (encoder, features)
            artifact_path = client.download_artifacts(latest_version.run_id, "")
            encoder = joblib.load(f"{artifact_path}/atm_encoder.pkl")
            features = joblib.load(f"{artifact_path}/features.pkl")
            
            model_cache.update({
                "model": model,
                "version": latest_version.version,
                "features": features,
                "encoder": encoder,
                "loaded_at": datetime.now()
            })
            
            print(f"✅ Modèle v{latest_version.version} chargé")
        
        return model_cache
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur chargement modèle: {str(e)}")


def enrich_features(req: PredictionRequest, encoder) -> pd.DataFrame:
    """Enrichir les features depuis la requête"""
    dt = datetime.fromisoformat(req.datetime)
    
    # Features temporelles de base
    hour = dt.hour
    dayofweek = dt.weekday()
    month = dt.month
    
    # Encoder atm
    atm_encoded = encoder.transform([req.atm])[0] if encoder else req.atm
    
    # Features calculées
    data = {
        'latitude': req.latitude,
        'longitude': req.longitude,
        'hour': hour,
        'dayofweek': dayofweek,
        'month': month,
        'atm': atm_encoded,
        
        # Interactions spatio-temporelles
        'lat_hour': req.latitude * hour / 24,
        'lon_hour': req.longitude * hour / 24,
        'lat_dayofweek': req.latitude * dayofweek / 7,
        'lon_dayofweek': req.longitude * dayofweek / 7,
        
        # Features binaires
        'is_weekend': int(dayofweek >= 5),
        'is_rush_morning': int(6 <= hour <= 8),
        'is_rush_evening': int(16 <= hour <= 18),
        'is_night': int(hour >= 19 or hour <= 5),
        
        # Encodage cyclique
        'hour_sin': np.sin(2 * np.pi * hour / 24),
        'hour_cos': np.cos(2 * np.pi * hour / 24),
        'dayofweek_sin': np.sin(2 * np.pi * dayofweek / 7),
        'dayofweek_cos': np.cos(2 * np.pi * dayofweek / 7),
        
        # Features OSM + densité (valeurs par défaut si pas de grille)
        'road_type': 2,  # Route secondaire par défaut
        'speed_limit': 50,
        'accident_density_5km': 0,
        'dist_to_noumea_km': 100,
        
        # Features temporelles avancées
        'is_holiday': 0,  # Simplification
        'school_holidays': int(month in [1, 7, 8, 12])
    }
    
    return pd.DataFrame([data])


def categorize_risk(score: float) -> str:
    """Catégoriser le niveau de risque"""
    if score < 0.2:
        return "LOW"
    elif score < 0.5:
        return "MEDIUM"
    elif score < 0.8:
        return "HIGH"
    else:
        return "CRITICAL"


@app.on_event("startup")
async def startup_event():
    """Charger le modèle au démarrage"""
    print("🚀 Démarrage de l'API...")
    load_production_model()
    print("✅ API prête")


@app.get("/health")
async def health():
    """Vérifier l'état de l'API"""
    cache = model_cache
    return {
        "status": "healthy" if cache["model"] else "initializing",
        "model_version": cache["version"],
        "model_loaded_at": cache["loaded_at"].isoformat() if cache["loaded_at"] else None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    """Prédire le risque d'accident"""
    # Charger le modèle (avec cache)
    cache = load_production_model()
    model = cache["model"]
    encoder = cache["encoder"]
    features_list = cache["features"]
    
    # Enrichir les features
    features_df = enrich_features(req, encoder)
    
    # S'assurer d'avoir toutes les features attendues
    features_df = features_df[features_list]
    
    # Prédiction
    risk_score = float(model.predict_proba(features_df)[0][1])
    
    return PredictionResponse(
        risk_score=risk_score,
        risk_level=categorize_risk(risk_score),
        confidence=max(risk_score, 1 - risk_score),
        model_version=cache["version"],
        features_used={
            "latitude": req.latitude,
            "longitude": req.longitude,
            "datetime": req.datetime,
            "hour": features_df['hour'].values[0],
            "is_weekend": bool(features_df['is_weekend'].values[0])
        }
    )


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(requests: List[PredictionRequest]):
    """Prédictions en batch"""
    return [await predict(req) for req in requests]


@app.post("/reload-model")
async def reload_model():
    """Forcer le rechargement du modèle (pour déploiement sans redémarrage)"""
    model_cache["version"] = None  # Invalider le cache
    load_production_model()
    return {"status": "reloaded", "version": model_cache["version"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
