"""Classic ML prediction endpoint: the typical "serve a trained model" pattern."""
from fastapi import APIRouter

from app.api.deps import IrisDep
from app.schemas.predict import IrisFeatures, IrisPrediction

router = APIRouter(prefix="/predict", tags=["classic ML"])


@router.post("/iris", response_model=IrisPrediction)
def predict_iris(features: IrisFeatures, model: IrisDep) -> IrisPrediction:
    """Predict a flower species from 4 measurements (cm)."""
    probabilities = model.predict(**features.model_dump())
    species = max(probabilities, key=probabilities.get)
    return IrisPrediction(species=species, confidence=probabilities[species],
                          probabilities=probabilities)
