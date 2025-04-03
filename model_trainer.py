import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import joblib
import os
from typing import Dict, Any, Tuple
from .data_processor import DataProcessor

class ModelTrainer:
    def __init__(self, model_dir: str = "models"):
        """Initialize the model trainer with a directory for saving models."""
        self.model_dir = model_dir
        self.data_processor = DataProcessor()
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def train_model(self, df: pd.DataFrame = None, model_type: str = "random_forest") -> Dict[str, Any]:
        """Train a model on the given data or generate synthetic data if None."""
        # If no data provided, generate synthetic data
        if df is None:
            df = self.data_processor.generate_sample_data()
        
        # Split data into features and target
        X, y = self.data_processor.preprocess_data(df)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save the scaler
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        
        # Select and train model
        if model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "gradient_boosting":
            model = GradientBoostingRegressor(random_state=42)
        elif model_type == "linear":
            model = LinearRegression()
        elif model_type == "ridge":
            model = Ridge(alpha=1.0)
        elif model_type == "lasso":
            model = Lasso(alpha=0.1)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Save the model
        model_path = os.path.join(self.model_dir, f"{model_type}_model.pkl")
        joblib.dump(model, model_path)
        
        # Evaluate the model
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred)
        }
        
        # Return results
        results = {
            "model_type": model_type,
            "model_path": model_path,
            "scaler_path": scaler_path,
            "metrics": metrics
        }
        
        return results
    
    def tune_model(self, df: pd.DataFrame = None, model_type: str = "random_forest") -> Dict[str, Any]:
        """Tune a model on the given data."""
        # If no data provided, generate synthetic data
        if df is None:
            df = self.data_processor.generate_sample_data()
        
        # Split data into features and target
        X, y = self.data_processor.preprocess_data(df)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define the parameter grid based on model type
        if model_type == "random_forest":
            model = RandomForestRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        elif model_type == "gradient_boosting":
            model = GradientBoostingRegressor(random_state=42)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7]
            }
        else:
            raise ValueError(f"Tuning not implemented for model type: {model_type}")
        
        # Perform grid search
        grid_search = GridSearchCV(
            model,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Get the best model
        best_model = grid_search.best_estimator_
        
        # Save the scaler
        scaler_path = os.path.join(self.model_dir, f"tuned_{model_type}_scaler.pkl")
        joblib.dump(scaler, scaler_path)
        
        # Save the best model
        model_path = os.path.join(self.model_dir, f"tuned_{model_type}_model.pkl")
        joblib.dump(best_model, model_path)
        
        # Evaluate the best model
        y_pred = best_model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred)
        }
        
        # Return results
        results = {
            "model_type": f"tuned_{model_type}",
            "model_path": model_path,
            "scaler_path": scaler_path,
            "best_params": grid_search.best_params_,
            "metrics": metrics
        }
        
        return results