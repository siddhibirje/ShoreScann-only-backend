import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import joblib
import os

class DataProcessor:
    def __init__(self, scaler_path: str = None):
        """Initialize the data processor with an optional path to a saved scaler."""
        self.scaler = None
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
    
    def generate_sample_data(self, start_year: int = 2000, end_year: int = 2024) -> pd.DataFrame:
        """Generate synthetic Mumbai coastal erosion data."""
        years = list(range(start_year, end_year + 1))
        months = list(range(1, 13))
        
        data = []
        for year in years:
            for month in months:
                # Seasonal patterns
                is_monsoon = 1 if month in [6, 7, 8, 9] else 0
                
                # Mumbai's rainfall pattern (monsoon-heavy)
                base_rainfall = np.random.normal(50, 20)
                rainfall = base_rainfall
                if is_monsoon:
                    rainfall += np.random.normal(300, 100)  # Heavy monsoon rainfall
                rainfall = max(0, rainfall)  # No negative rainfall
                
                # Temperature pattern for Mumbai
                temp_base = 24 + 5 * np.sin(2 * np.pi * (month - 1) / 12)  # Seasonal variation
                temperature = temp_base + np.random.normal(0, 1.5)  # Add some noise
                
                # Tidal range - higher during monsoons and full moons
                tidal_range_base = 2.5 + 0.5 * np.sin(2 * np.pi * (month - 1) / 12)
                tidal_range = tidal_range_base + np.random.normal(0, 0.3)
                
                # Create a time trend factor (increasing erosion over years)
                time_trend = 0.05 * (year - start_year)
                
                # Calculate erosion rate with some realistic dependencies
                erosion_rate = (
                    0.5 +  # Base erosion rate
                    0.002 * rainfall +  # Effect of rainfall
                    0.3 * tidal_range +  # Effect of tidal range
                    0.05 * temperature +  # Effect of temperature
                    time_trend +  # Long-term trend
                    np.random.normal(0, 0.5)  # Random variation
                )
                
                # Add some year-to-year variation
                if year > start_year:
                    year_effect = np.sin(year * 0.4) * 0.3
                    erosion_rate += year_effect
                    
                # Ensure no negative erosion
                erosion_rate = max(0, erosion_rate)
                
                data.append({
                    'Year': year,
                    'Month': month,
                    'Rainfall': round(rainfall, 1),
                    'Temperature': round(temperature, 1),
                    'Tidal_Range': round(tidal_range, 2),
                    'Erosion_Rate': round(erosion_rate, 2),
                    'Is_Monsoon': is_monsoon
                })
        
        return pd.DataFrame(data)
    
    def preprocess_data(self, df: pd.DataFrame, for_training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the data for model training or inference."""
        # Create engineered features
        df['TimeIndex'] = df['Year'] + df['Month']/12 - 2000
        df['Month_sin'] = np.sin(2 * np.pi * df['Month']/12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month']/12)
        
        # Extract features and target
        features = ['TimeIndex', 'Rainfall', 'Temperature', 'Tidal_Range', 'Month_sin', 'Month_cos', 'Is_Monsoon']
        X = df[features].values
        
        if for_training:
            y = df['Erosion_Rate'].values
            return X, y
        return X, None
    
    def generate_future_data(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Generate data for future predictions."""
        future_years = list(range(start_year, end_year + 1))
        future_data = []

        # Use historical averages 
        monthly_rainfall_mean = {
            1: 10.2, 2: 5.6, 3: 8.4, 4: 12.1, 5: 18.3, 
            6: 320.5, 7: 580.2, 8: 520.4, 9: 312.6, 
            10: 85.3, 11: 25.6, 12: 12.3
        }
        
        monthly_temp_mean = {
            1: 24.1, 2: 24.8, 3: 26.7, 4: 28.6, 5: 30.5, 
            6: 29.8, 7: 28.2, 8: 27.9, 9: 28.4, 
            10: 28.9, 11: 27.6, 12: 25.8
        }
        
        monthly_tidal_mean = {
            1: 2.8, 2: 2.7, 3: 2.6, 4: 2.5, 5: 2.6, 
            6: 2.9, 7: 3.1, 8: 3.2, 9: 3.1, 
            10: 3.0, 11: 2.9, 12: 2.8
        }

        for year in future_years:
            for month in range(1, 13):
                # Climate change effects
                year_factor = (year - 2024) * 0.01  # 1% increase per year
                
                # Generate reasonable values
                is_monsoon = 1 if month in [6, 7, 8, 9] else 0
                
                # Add randomness and climate change trends
                rainfall = monthly_rainfall_mean[month] * (1 + year_factor) + np.random.normal(0, monthly_rainfall_mean[month] * 0.1)
                temperature = monthly_temp_mean[month] + (year - 2024) * 0.03 + np.random.normal(0, 0.5)
                tidal_range = monthly_tidal_mean[month] + (year - 2024) * 0.01 + np.random.normal(0, 0.05)
                
                # Ensure values are reasonable
                rainfall = max(0, rainfall)
                
                future_data.append({
                    'Year': year,
                    'Month': month,
                    'Rainfall': round(rainfall, 1),
                    'Temperature': round(temperature, 1),
                    'Tidal_Range': round(tidal_range, 2),
                    'Is_Monsoon': is_monsoon
                })

        return pd.DataFrame(future_data)