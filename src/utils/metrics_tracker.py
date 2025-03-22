"""
Utility to track and compare model performance metrics across runs
"""

import os
import json
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

class MetricsTracker:
    """Class to track model performance metrics across runs"""
    
    def __init__(self, metrics_file: str = 'metrics_history.json'):
        """
        Initialize metrics tracker
        
        Args:
            metrics_file (str): Path to metrics history file
        """
        self.metrics_file = metrics_file
        self.metrics_history = self._load_metrics()
    
    def _load_metrics(self) -> List[Dict[str, Any]]:
        """
        Load metrics history from file
        
        Returns:
            list: List of metrics dictionaries
        """
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []
        return []
    
    def _save_metrics(self) -> None:
        """Save metrics history to file"""
        os.makedirs(os.path.dirname(self.metrics_file) or '.', exist_ok=True)
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def add_metrics(self, metrics: Dict[str, Any], 
                    model_params: Optional[Dict[str, Any]] = None,
                    run_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a new set of metrics to the history
        
        Args:
            metrics (dict): Performance metrics (accuracy, precision, etc.)
            model_params (dict, optional): Model parameters
            run_params (dict, optional): Run parameters
        """
        # Create new metrics entry
        entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'metrics': metrics
        }
        
        # Add model parameters if provided
        if model_params:
            entry['model_params'] = model_params
        
        # Add run parameters if provided
        if run_params:
            entry['run_params'] = run_params
        
        # Add to history
        self.metrics_history.append(entry)
        
        # Save updated history
        self._save_metrics()
    
    def get_metrics(self) -> pd.DataFrame:
        """
        Get metrics history as DataFrame
        
        Returns:
            pd.DataFrame: Metrics history
        """
        if not self.metrics_history:
            return pd.DataFrame()
        
        # Create list of rows for DataFrame
        rows = []
        for entry in self.metrics_history:
            row = {'timestamp': entry['timestamp']}
            
            # Add metrics
            for metric, value in entry['metrics'].items():
                row[f'metric_{metric}'] = value
            
            # Add model parameters
            if 'model_params' in entry:
                for param, value in entry['model_params'].items():
                    row[f'param_{param}'] = value
            
            # Add run parameters
            if 'run_params' in entry:
                for param, value in entry['run_params'].items():
                    row[f'run_{param}'] = value
            
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def plot_metric(self, metric: str, figsize: tuple = (10, 6)) -> None:
        """
        Plot a specific metric over time
        
        Args:
            metric (str): Metric name
            figsize (tuple): Figure size
        """
        # Get metrics DataFrame
        df = self.get_metrics()
        if df.empty:
            print("No metrics history available.")
            return
        
        # Check if metric exists
        metric_col = f'metric_{metric}'
        if metric_col not in df.columns:
            print(f"Metric '{metric}' not found in history.")
            return
        
        # Create plot
        plt.figure(figsize=figsize)
        plt.plot(df['timestamp'], df[metric_col], 'o-')
        plt.title(f'{metric} over time')
        plt.xlabel('Date')
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def compare_models(self, metric: str, param: str, figsize: tuple = (10, 6)) -> None:
        """
        Compare models based on a parameter
        
        Args:
            metric (str): Metric to compare
            param (str): Parameter to group by
            figsize (tuple): Figure size
        """
        # Get metrics DataFrame
        df = self.get_metrics()
        if df.empty:
            print("No metrics history available.")
            return
        
        # Check if metric and parameter exist
        metric_col = f'metric_{metric}'
        param_col = f'param_{param}'
        
        if metric_col not in df.columns:
            print(f"Metric '{metric}' not found in history.")
            return
        
        if param_col not in df.columns:
            print(f"Parameter '{param}' not found in history.")
            return
        
        # Group by parameter and get mean of metric
        grouped = df.groupby(param_col)[metric_col].mean().reset_index()
        
        # Create plot
        plt.figure(figsize=figsize)
        plt.bar(grouped[param_col].astype(str), grouped[metric_col])
        plt.title(f'{metric} by {param}')
        plt.xlabel(param)
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def get_best_model(self, metric: str, higher_is_better: bool = True) -> Dict[str, Any]:
        """
        Get the best model based on a metric
        
        Args:
            metric (str): Metric to use for comparison
            higher_is_better (bool): Whether higher metric values are better
            
        Returns:
            dict: Best model entry
        """
        # Get metrics DataFrame
        df = self.get_metrics()
        if df.empty:
            print("No metrics history available.")
            return {}
        
        # Check if metric exists
        metric_col = f'metric_{metric}'
        if metric_col not in df.columns:
            print(f"Metric '{metric}' not found in history.")
            return {}
        
        # Get index of best model
        if higher_is_better:
            best_idx = df[metric_col].idxmax()
        else:
            best_idx = df[metric_col].idxmin()
        
        # Get best model entry
        best_entry = df.iloc[best_idx].to_dict()
        
        # Format result
        result = {
            'timestamp': best_entry['timestamp'],
            'metrics': {},
            'model_params': {},
            'run_params': {}
        }
        
        # Add metrics
        for col in df.columns:
            if col.startswith('metric_'):
                metric_name = col[len('metric_'):]
                result['metrics'][metric_name] = best_entry[col]
            elif col.startswith('param_'):
                param_name = col[len('param_'):]
                result['model_params'][param_name] = best_entry[col]
            elif col.startswith('run_'):
                param_name = col[len('run_'):]
                result['run_params'][param_name] = best_entry[col]
        
        return result
