"""
Base model class for trading entry point prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

class BaseModel:
    """Base model for trading entry point prediction"""
    
    def __init__(self, random_state=42):
        """
        Initialize the model
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.scaler = None
    
    def extract_features_target(self, df):
        """
        Extract features and target from prepared DataFrame
        
        Args:
            df (pd.DataFrame): DataFrame with features and target
            
        Returns:
            tuple: X (features), y (target)
        """
        # Exclude non-feature columns
        exclude_cols = ['Target', 'Future_Return', 'Open', 'High', 'Low', 'Close', 'Volume']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_names = feature_cols
        
        # Replace infinity values with NaN
        df_features = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with column means
        col_means = df_features.mean()
        df_features = df_features.fillna(col_means)
        
        # Extract features and target
        X = df_features.values
        y = df['Target'].values if 'Target' in df.columns else None
        
        return X, y
    
    def train(self, X, y):
        """
        Train the model
        
        Args:
            X (ndarray): Feature matrix
            y (ndarray): Target vector
            
        Returns:
            self: For method chaining
        """
        raise NotImplementedError("Subclasses must implement train method")
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X (ndarray): Feature matrix
            
        Returns:
            ndarray: Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X (ndarray): Feature matrix
            
        Returns:
            ndarray: Class probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.model.predict_proba(X)
    
    def evaluate(self, X, y):
        """
        Evaluate model performance
        
        Args:
            X (ndarray): Feature matrix
            y (ndarray): True target values
            
        Returns:
            dict: Performance metrics
        """
        y_pred = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y, y_pred, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def explain_prediction(self, features):
        """
        Generate explanation for a prediction
        
        Args:
            features (ndarray): Feature vector (1D array or matrix with 1 row)
            
        Returns:
            dict: Explanation details
        """
        raise NotImplementedError("Subclasses must implement explain_prediction method")
    
    def save(self, filepath):
        """
        Save model to file
        
        Args:
            filepath (str): Path to save the model
            
        Returns:
            str: Path to saved model
        """
        if self.model is None:
            raise ValueError("No trained model to save.")
            
        # Create a dictionary with everything needed to restore the model
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'random_state': self.random_state,
            'scaler': self.scaler if hasattr(self, 'scaler') else None
        }
        
        joblib.dump(model_data, filepath)
        return filepath
    
    @classmethod
    def load(cls, filepath):
        """
        Load model from file
        
        Args:
            filepath (str): Path to saved model
            
        Returns:
            BaseModel: Loaded model instance
        """
        model_data = joblib.load(filepath)
        
        # Create a new instance
        instance = cls(random_state=model_data.get('random_state', 42))
        
        # Restore model attributes
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.scaler = model_data.get('scaler')
        
        return instance
