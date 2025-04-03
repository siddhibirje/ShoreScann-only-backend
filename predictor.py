import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Any, Union
from .data_processor import DataProcessor
import os

class ErosionPredictor:
    def __init__(self, model_path: str, scaler_path: str):
        """Initialize the erosion predictor with model and scaler paths."""
        self.model = None
        self.scaler = None
        self.data_processor = DataProcessor()
        
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    
    def predict_erosion(self, data: Dict[str, Any]) -> float:
        """Predict erosion rate from single data point."""
        # Convert single data point to DataFrame
        df = pd.DataFrame([data])
        
        # Preprocess the data
        X, _ = self.data_processor.preprocess_data(df, for_training=False)
        
        # Scale the features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        
        return float(prediction)
    
    def predict_batch(self, data: List[Dict[str, Any]]) -> List[float]:
        """Predict erosion rates for multiple data points."""
        # Convert list of data points to DataFrame
        df = pd.DataFrame(data)
        
        # Preprocess the data
        X, _ = self.data_processor.preprocess_data(df, for_training=False)
        
        # Scale the features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled).tolist()
        
        return predictions
    
    def predict_future(self, start_year: int = 2025, end_year: int = 2030) -> pd.DataFrame:
        """Generate and predict future erosion rates."""
        # Generate future data
        future_df = self.data_processor.generate_future_data(start_year, end_year)
        
        # Preprocess the data
        X, _ = self.data_processor.preprocess_data(future_df, for_training=False)
        
        # Scale the features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Add predictions to the DataFrame
        future_df['Predicted_Erosion'] = predictions
        
        return future_df