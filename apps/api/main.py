from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .inference import inference_service
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelMetadata,
    PredictionRequest,
    PredictionResponse,
)

app = FastAPI(
    title="Accidents NC - API d'Inférence",
    description="API de prédiction de gravité des accidents en Nouvelle-Calédonie",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    return {
        "message": "Bienvenue sur l'API Accidents NC",
        "status": "ready" if inference_service.is_ready else "loading",
        "docs": "/docs",
    }


@app.get("/health", tags=["General"])
async def health():
    if not inference_service.is_ready:
        inference_service.load_artifacts()

    return {
        "status": "ok" if inference_service.is_ready else "error",
        "models_loaded": list(inference_service.models.keys()),
    }


@app.get("/metadata", response_model=ModelMetadata, tags=["Model"])
async def get_metadata():
    if not inference_service.is_ready:
        raise HTTPException(status_code=503, detail="Service non prêt")

    return ModelMetadata(
        name="Blending Ensemble (CatBoost, XGBoost, MLP)",
        version="1.0.0",
        features=inference_service.features,
        threshold=inference_service.threshold,
        trained_at="2026-03-06",  # Devrait être dynamique
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(request: PredictionRequest):
    if not inference_service.is_ready:
        inference_service.load_artifacts()

    if not inference_service.is_ready:
        raise HTTPException(status_code=503, detail="Modèles non chargés")

    try:
        return inference_service.predict(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(request: BatchPredictionRequest):
    if not inference_service.is_ready:
        inference_service.load_artifacts()

    if not inference_service.is_ready:
        raise HTTPException(status_code=503, detail="Service non prêt")

    try:
        return inference_service.predict_batch(request)
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e
