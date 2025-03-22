"""
Random Forest model for trading entry point prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    """Random Forest model for trading entry point prediction"""
    
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=10, 
                 random_state=42, profit_target=0.01, stop_loss=0.005):
        """
        Initialize Random Forest model
        
        Args:
            n_estimators (int): Number of trees in the forest
            max_depth (int): Maximum depth of trees
            min_samples_split (int): Minimum samples required to split a node
            random_state (int): Random seed for reproducibility
            profit_target (float): Target profit as a decimal (e.g., 0.01 = 1%)
            stop_loss (float): Stop loss as a decimal (e.g., 0.005 = 0.5%)
        """
        super().__init__(random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.profit_target = profit_target
        self.stop_loss = stop_loss
    
    def train(self, X, y):
        """
        Train Random Forest model
        
        Args:
            X (ndarray): Feature matrix
            y (ndarray): Target vector
            
        Returns:
            self: For method chaining
        """
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state
        )
        
        self.model.fit(X, y)
        return self
    
    def get_feature_importance(self):
        """
        Get feature importance from the trained model
        
        Returns:
            pd.DataFrame: Feature importance DataFrame sorted by importance
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        importances = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        return feature_importance
    
    def explain_prediction(self, features, prediction=None, probabilities=None):
        """
        Generate natural language explanation for a prediction
        
        Args:
            features (ndarray): Feature vector (1D array or matrix with 1 row)
            prediction (int, optional): Prediction to explain (if None, will be predicted)
            probabilities (ndarray, optional): Prediction probabilities (if None, will be calculated)
            
        Returns:
            dict: Explanation details
        """
        # Reshape features if needed
        if features.ndim == 1:
            features_reshaped = features.reshape(1, -1)
        else:
            features_reshaped = features
        
        # Get prediction and probabilities if not provided
        if prediction is None:
            prediction = self.predict(features_reshaped)[0]
        
        if probabilities is None:
            probabilities = self.predict_proba(features_reshaped)[0]
        
        # Map prediction to direction and get confidence based on probabilities structure
        # Handle different types of probability structures
        if isinstance(probabilities, (float, np.float64, np.float32)):
            # If probabilities is a single value, use it directly
            confidence = probabilities
            direction = "bullish movement" if prediction == 1 else "bearish movement" if prediction == -1 else "no significant move"
        elif hasattr(probabilities, 'ndim') and probabilities.ndim == 1:
            # If probabilities is a 1D array
            if prediction == 0:
                direction = "no significant move"
                confidence = probabilities[0] if len(probabilities) > 0 else 0.5
            elif prediction == 1:
                direction = "bullish movement"
                confidence = probabilities[1] if len(probabilities) > 1 else probabilities[0]
            else:  # -1
                direction = "bearish movement"
                confidence = probabilities[2] if len(probabilities) > 2 else probabilities[0]
        else:
            # If probabilities is a 2D array or list of lists
            probs_row = probabilities[0] if hasattr(probabilities, '__getitem__') else probabilities
            if prediction == 0:
                direction = "no significant move"
                confidence = probs_row[0] if hasattr(probs_row, '__getitem__') and len(probs_row) > 0 else 0.5
            elif prediction == 1:
                direction = "bullish movement"
                confidence = probs_row[1] if hasattr(probs_row, '__getitem__') and len(probs_row) > 1 else probs_row[0]
            else:  # -1
                direction = "bearish movement"
                confidence = probs_row[2] if hasattr(probs_row, '__getitem__') and len(probs_row) > 2 else probs_row[0]
        
        # Get top influential features
        feature_importance = self.get_feature_importance()
        top_features = feature_importance.head(5)['Feature'].values
        
        # Create explanation
        explanation = f"Predicted {direction} with {confidence:.1%} confidence based on:\n"
        
        # Add feature-specific reasoning
        for i, feature in enumerate(top_features):
            feature_idx = self.feature_names.index(feature)
            feature_value = features[feature_idx]
            
            # Add context based on feature type
            if 'RSI' in feature:
                if feature_value > 70:
                    context = f"RSI is overbought at {feature_value:.2f}"
                elif feature_value < 30:
                    context = f"RSI is oversold at {feature_value:.2f}"
                else:
                    context = f"RSI is neutral at {feature_value:.2f}"
            elif 'MACD' in feature:
                if 'Cross_Above' in feature and feature_value == 1:
                    context = "MACD crossed above signal line"
                elif 'Cross_Below' in feature and feature_value == 1:
                    context = "MACD crossed below signal line"
                else:
                    context = f"MACD indicator value is {feature_value:.2f}"
            elif 'BB_Position' in feature:
                if feature_value > 0.8:
                    context = f"Price is near upper Bollinger Band ({feature_value:.2f})"
                elif feature_value < 0.2:
                    context = f"Price is near lower Bollinger Band ({feature_value:.2f})"
                else:
                    context = f"Price is within Bollinger Bands ({feature_value:.2f})"
            elif 'SMA' in feature or 'EMA' in feature:
                if 'Ratio' in feature:
                    if feature_value > 1.02:
                        context = f"Price is significantly above {feature.split('_')[1]} ({feature_value:.2f})"
                    elif feature_value < 0.98:
                        context = f"Price is significantly below {feature.split('_')[1]} ({feature_value:.2f})"
                    else:
                        context = f"Price is near {feature.split('_')[1]} ({feature_value:.2f})"
                else:
                    context = f"{feature} value is {feature_value:.2f}"
            elif 'ATR' in feature:
                context = f"Market volatility (ATR) is {feature_value:.2f}"
            else:
                context = f"{feature} value is {feature_value:.2f}"
            
            explanation += f"- {context}\n"
        
        # Add trading recommendation
        if prediction == 1:
            explanation += f"\nRecommendation: Consider LONG entry with target at +{self.profit_target*100:.1f}% and stop loss at -{self.stop_loss*100:.1f}%"
        elif prediction == -1:
            explanation += f"\nRecommendation: Consider SHORT entry with target at -{self.profit_target*100:.1f}% and stop loss at +{self.stop_loss*100:.1f}%"
        else:
            explanation += "\nRecommendation: Stay on the sidelines, no clear trading opportunity"
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'direction': direction,
            'top_features': top_features,
            'explanation': explanation
        }
    
    def find_entry_points(self, X, dates, prices, confidence_threshold=0.6):
        """
        Find potential entry points with high confidence
        
        Args:
            X (ndarray): Feature matrix
            dates (array-like): Corresponding dates for each row in X
            prices (array-like): Corresponding prices for each row in X
            confidence_threshold (float): Minimum confidence required for entry point
            
        Returns:
            list: Entry points with details
        """
        # Get predictions and probabilities
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        # Create entry points list
        entry_points = []
        
        for i in range(len(predictions)):
            pred = predictions[i]
            
            # Skip neutral predictions
            if pred == 0:
                continue
            
            # Get confidence
            if pred == 1:
                confidence = probabilities[i][1] if probabilities[i].shape[0] > 1 else probabilities[i][0]
            else:  # pred == -1
                confidence = probabilities[i][2] if probabilities[i].shape[0] > 2 else probabilities[i][0]
            
            # Skip low confidence predictions
            if confidence < confidence_threshold:
                continue
            
            # Create entry point
            price = prices[i]
            entry_point = {
                'date': dates[i],
                'price': price,
                'direction': 'LONG' if pred == 1 else 'SHORT',
                'confidence': confidence,
                'target_price': price * (1 + self.profit_target if pred == 1 else 1 - self.profit_target),
                'stop_loss_price': price * (1 - self.stop_loss if pred == 1 else 1 + self.stop_loss),
                'explanation': self.explain_prediction(X[i], pred, probabilities[i])['explanation']
            }
            
            entry_points.append(entry_point)
        
        return entry_points