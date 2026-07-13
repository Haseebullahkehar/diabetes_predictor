"""Request/response schemas for the diabetes risk API."""
from pydantic import BaseModel, Field


class PatientData(BaseModel):
    """Clinical inputs for a single patient. Ranges are sanity bounds, not medical limits."""

    Pregnancies: int = Field(..., ge=0, le=20, description="Number of times pregnant", examples=[2])
    Glucose: float = Field(..., ge=0, le=300, description="Plasma glucose (mg/dL, 2h OGTT)", examples=[120])
    BloodPressure: float = Field(..., ge=0, le=200, description="Diastolic blood pressure (mm Hg)", examples=[70])
    SkinThickness: float = Field(..., ge=0, le=100, description="Triceps skin fold thickness (mm)", examples=[20])
    Insulin: float = Field(..., ge=0, le=900, description="2-Hour serum insulin (mu U/ml)", examples=[79])
    BMI: float = Field(..., ge=0, le=70, description="Body mass index (kg/m^2)", examples=[28.5])
    DiabetesPedigreeFunction: float = Field(
        ..., ge=0, le=3, description="Diabetes pedigree (family history score)", examples=[0.45]
    )
    Age: int = Field(..., ge=1, le=120, description="Age in years", examples=[33])


class PredictionResponse(BaseModel):
    probability: float = Field(..., description="Predicted probability of diabetes (0-1)")
    risk_percent: float = Field(..., description="Probability expressed as a percentage")
    risk_level: str = Field(..., description="Low / Moderate / High")
    prediction: int = Field(..., description="1 = likely diabetic, 0 = not, at the optimized threshold")
    threshold: float = Field(..., description="Decision threshold used to produce the 0/1 prediction")
    model_name: str = Field(..., description="Algorithm used for the prediction")
