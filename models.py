from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime

class ErosionDataPoint(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    Year: int = Field(..., ge=2000, le=2050)
    Month: int = Field(..., ge=1, le=12)
    Rainfall: float = Field(..., ge=0)
    Temperature: float = Field(..., ge=0)
    Tidal_Range: float = Field(..., ge=0)
    Is_Monsoon: Optional[int] = Field(None, ge=0, le=1)

class ErosionPredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    erosion_rate: float
    timestamp: datetime = Field(default_factory=datetime.now)

class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    data_points: List[ErosionDataPoint]

class BatchPredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    predictions: List[float]
    timestamp: datetime = Field(default_factory=datetime.now)

class FuturePredictionRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    start_year: int = Field(..., ge=2025, le=2050)
    end_year: int = Field(..., ge=2025, le=2050)

class FuturePrediction(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    Year: int
    Month: int
    Rainfall: float
    Temperature: float
    Tidal_Range: float
    Is_Monsoon: int
    Predicted_Erosion: float

class FuturePredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    predictions: List[FuturePrediction]
    timestamp: datetime = Field(default_factory=datetime.now)

class ModelTrainingRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    model_type: str = Field("random_forest", description="Type of model to train")
    tune_model: bool = Field(False, description="Whether to tune the model")

class ModelTrainingResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    model_type: str
    model_path: str
    metrics: Dict[str, float]
    timestamp: datetime = Field(default_factory=datetime.now)