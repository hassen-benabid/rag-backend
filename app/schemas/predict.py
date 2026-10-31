from pydantic import BaseModel, Field


class IrisFeatures(BaseModel):
    # gt = greater than, le = less or equal. Invalid values are rejected before our code runs.
    sepal_length: float = Field(gt=0, le=20, description="cm", examples=[5.1])
    sepal_width: float = Field(gt=0, le=20, description="cm", examples=[3.5])
    petal_length: float = Field(gt=0, le=20, description="cm", examples=[1.4])
    petal_width: float = Field(gt=0, le=20, description="cm", examples=[0.2])


class IrisPrediction(BaseModel):
    species: str
    confidence: float
    probabilities: dict[str, float]
