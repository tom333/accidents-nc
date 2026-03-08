from datetime import datetime

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    latitude: float = Field(..., description="Latitude de l'accident potentiel")
    longitude: float = Field(..., description="Longitude de l'accident potentiel")
    timestamp: datetime | None = Field(
        default_factory=datetime.now, description="Date et heure de l'événement"
    )
    atm: str = Field(
        default="Normale",
        description="Conditions atmosphériques (ex: Normale, Pluie, Brouillard)",
    )


class PredictionResponse(BaseModel):
    probability: float = Field(..., description="Probabilité d'un accident grave")
    risk_level: str = Field(..., description="Niveau de risque (Faible, Modéré, Élevé)")
    threshold: float = Field(..., description="Seuil optimal utilisé pour la classification")
    is_dangerous: bool = Field(..., description="Indique si le risque dépasse le seuil")
    model_version: str = Field(..., description="Version du modèle utilisé")


class ModelMetadata(BaseModel):
    name: str
    version: str
    features: list[str]
    threshold: float
    trained_at: str


class BatchPredictionRequest(BaseModel):
    locations: list[dict] = Field(
        ..., description="Liste de dictionnaires avec 'latitude' et 'longitude'"
    )
    timestamp: datetime | None = Field(
        default_factory=datetime.now, description="Date et heure pour tout le lot"
    )
    atm: str = Field(default="Normale", description="Conditions atmosphériques pour tout le lot")


class BatchPredictionResponse(BaseModel):
    predictions: list[float] = Field(..., description="Liste des probabilités prédites")
    threshold: float = Field(..., description="Seuil utilisé")
    model_version: str = Field(..., description="Version du modèle")
