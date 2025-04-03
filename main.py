from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import os
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime

# Change relative imports to absolute imports
from shoreline_erosion_api.models import (
    ErosionDataPoint, 
    ErosionPredictionResponse, 
    BatchPredictionRequest, 
    BatchPredictionResponse,
    FuturePredictionRequest,
    FuturePredictionResponse,
    FuturePrediction,
    ModelTrainingRequest,
    ModelTrainingResponse
)
from shoreline_erosion_api.data_processor import DataProcessor
from shoreline_erosion_api.predictor import ErosionPredictor
from shoreline_erosion_api.model_trainer import ModelTrainer


# Create model directory if it doesn't exist
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Set default model paths
default_model_path = os.path.join(model_dir, "random_forest_model.pkl")
default_scaler_path = os.path.join(model_dir, "scaler.pkl")

# Initialize FastAPI app
app = FastAPI(
    title="Mumbai Coastal Erosion API",
    description="API for predicting coastal erosion in Mumbai",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
data_processor = DataProcessor()
model_trainer = ModelTrainer(model_dir=model_dir)

# Check if model exists, if not train a default model
if not os.path.exists(default_model_path):
    # Generate sample data
    sample_data = data_processor.generate_sample_data()
    # Train a default model
    training_results = model_trainer.train_model(sample_data)
    print(f"Trained default model: {training_results['model_type']}")

# Dependency to get predictor
def get_predictor():
    model_path = default_model_path
    scaler_path = default_scaler_path
    
    # Check if files exist
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise HTTPException(status_code=500, detail="Model files not found. Train a model first.")
    
    try:
        return ErosionPredictor(model_path=model_path, scaler_path=scaler_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Mumbai Coastal Erosion Prediction API"}

@app.post("/predict", response_model=ErosionPredictionResponse)
async def predict_erosion(data: ErosionDataPoint, predictor: ErosionPredictor = Depends(get_predictor)):
    try:
        # Convert Pydantic model to dict
        data_dict = data.dict()
        
        # Make prediction
        prediction = predictor.predict_erosion(data_dict)
        
        return ErosionPredictionResponse(
            erosion_rate=prediction,
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest, predictor: ErosionPredictor = Depends(get_predictor)):
    try:
        # Convert Pydantic models to dicts
        data_dicts = [point.dict() for point in request.data_points]
        
        # Make batch predictions
        predictions = predictor.predict_batch(data_dicts)
        
        return BatchPredictionResponse(
            predictions=predictions,
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.post("/predict/future", response_model=FuturePredictionResponse)
async def predict_future(request: FuturePredictionRequest, predictor: ErosionPredictor = Depends(get_predictor)):
    try:
        # Validate input
        if request.start_year > request.end_year:
            raise HTTPException(status_code=400, detail="Start year must be less than or equal to end year")
        
        # Generate and predict future data
        future_df = predictor.predict_future(request.start_year, request.end_year)
        
        # Convert DataFrame to list of Pydantic models
        predictions = []
        for _, row in future_df.iterrows():
            predictions.append(FuturePrediction(
                Year=int(row['Year']),
                Month=int(row['Month']),
                Rainfall=float(row['Rainfall']),
                Temperature=float(row['Temperature']),
                Tidal_Range=float(row['Tidal_Range']),
                Is_Monsoon=int(row['Is_Monsoon']),
                Predicted_Erosion=float(row['Predicted_Erosion'])
            ))
        
        return FuturePredictionResponse(
            predictions=predictions,
            timestamp=datetime.now()
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Future prediction error: {str(e)}")

@app.post("/train", response_model=ModelTrainingResponse)
async def train_model(request: ModelTrainingRequest):
    try:
        # Generate sample data
        sample_data = data_processor.generate_sample_data()
        
        # Train or tune the model
        if request.tune_model:
            results = model_trainer.tune_model(sample_data, model_type=request.model_type)
        else:
            results = model_trainer.train_model(sample_data, model_type=request.model_type)
        
        # Update default model paths if training was successful
        global default_model_path, default_scaler_path
        default_model_path = results['model_path']
        default_scaler_path = results['scaler_path']
        
        return ModelTrainingResponse(
            model_type=results['model_type'],
            model_path=results['model_path'],
            metrics=results['metrics'],
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}