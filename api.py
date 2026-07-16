"""
FastAPI backend for real-time child malnutrition risk prediction.
Run with: uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import MalnutritionPredictor

app = FastAPI(
    title="Child Malnutrition Risk Prediction API",
    description="Real-time multi-task deep learning risk screening for stunting, wasting, and underweight, "
                "with MC Dropout uncertainty and SHAP explainability. Built on DHS survey data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

predictor = MalnutritionPredictor()


class ChildRecord(BaseModel):
    age_months: float = Field(..., ge=0, le=59, description="Child's age in months (0-59)")
    sex: int = Field(..., ge=0, le=1, description="0 = male, 1 = female")
    wealth_index: int = Field(..., ge=1, le=5, description="Household wealth quintile (1=poorest, 5=richest)")
    residence_type: int = Field(..., ge=1, le=2, description="1 = urban, 2 = rural")
    mother_education: int = Field(..., ge=0, le=3, description="0=none,1=primary,2=secondary,3=higher")
    mother_age: float = Field(..., ge=12, le=55, description="Mother's age in years")
    region: str = Field(..., description="Region code/name as encoded in the training data")

    class Config:
        json_schema_extra = {
            "example": {
                "age_months": 18, "sex": 0, "wealth_index": 1,
                "residence_type": 2, "mother_education": 0,
                "mother_age": 22, "region": "1",
            }
        }


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Child Malnutrition Risk Prediction API is running.",
        "docs": "/docs",
    }


@app.get("/regions")
def get_valid_regions():
    """Returns the list of region codes the model was trained on."""
    return {"regions": [str(r) for r in predictor.region_encoder.classes_]}


@app.post("/predict")
def predict(record: ChildRecord, mc_samples: int = 30, explain: bool = True):
    try:
        result = predictor.predict(record.model_dump(), mc_samples=mc_samples, explain=explain)
        return {"input": record.model_dump(), "prediction": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
