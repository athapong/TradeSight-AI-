"""
Helper functions for the trading project
"""

import os
import pandas as pd
import numpy as np
import datetime
import json
import joblib

def set_pandas_display_options():
    """Set pandas display options for better notebook output"""
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.precision', 4)
    pd.set_option('display.float_format', '{:.4f}'.format)

def create_output_directory(base_dir='results'):
    """
    Create timestamped output directory for storing results
    
    Args:
        base_dir (str): Base directory name
        
    Returns:
        str: Path to created directory
    """
    # Create base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create timestamped directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_dir, f'run_{timestamp}')
    os.makedirs(output_dir)
    
    return output_dir

def save_results(output_dir, model=None, feature_importance=None, entry_points=None, 
                 backtest_results=None, backtest_analysis=None, figures=None):
    """
    Save all results to output directory
    
    Args:
        output_dir (str): Path to output directory
        model: Trained model object
        feature_importance (pd.DataFrame): Feature importance DataFrame
        entry_points (list): List of entry point dictionaries
        backtest_results (list): List of backtest result dictionaries
        backtest_analysis (dict): Backtest analysis results
        figures (dict): Dictionary of figure objects with names as keys
    """
    # Save model
    if model is not None:
        model_path = os.path.join(output_dir, 'model.pkl')
        joblib.dump(model, model_path)
    
    # Save feature importance
    if feature_importance is not None:
        feature_path = os.path.join(output_dir, 'feature_importance.csv')
        feature_importance.to_csv(feature_path, index=False)
    
    # Save entry points
    if entry_points is not None:
        entry_path = os.path.join(output_dir, 'entry_points.json')
        # Convert dates to strings for JSON serialization
        entry_points_json = []
        for entry in entry_points:
            entry_json = entry.copy()
            if isinstance(entry_json['date'], (pd.Timestamp, datetime.datetime)):
                entry_json['date'] = entry_json['date'].strftime('%Y-%m-%d %H:%M:%S')
            entry_points_json.append(entry_json)
        
        with open(entry_path, 'w') as f:
            json.dump(entry_points_json, f, indent=2)
    
    # Save backtest results
    if backtest_results is not None:
        backtest_path = os.path.join(output_dir, 'backtest_results.json')
        # Convert dates to strings for JSON serialization
        backtest_results_json = []
        for result in backtest_results:
            result_json = result.copy()
            if isinstance(result_json['timestamp'], (pd.Timestamp, datetime.datetime)):
                result_json['timestamp'] = result_json['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            if isinstance(result_json['exit_timestamp'], (pd.Timestamp, datetime.datetime)):
                result_json['exit_timestamp'] = result_json['exit_timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            backtest_results_json.append(result_json)
        
        with open(backtest_path, 'w') as f:
            json.dump(backtest_results_json, f, indent=2)
    
    # Save backtest analysis
    if backtest_analysis is not None:
        analysis_path = os.path.join(output_dir, 'backtest_analysis.json')
        with open(analysis_path, 'w') as f:
            # Handle non-serializable data types (like numpy types and infinity)
            analysis_serializable = {}
            for k, v in backtest_analysis.items():
                if isinstance(v, (np.integer, np.floating)):
                    analysis_serializable[k] = float(v)
                elif v == float('inf'):
                    analysis_serializable[k] = "Infinity"
                else:
                    analysis_serializable[k] = v
            
            json.dump(analysis_serializable, f, indent=2)
    
    # Save figures
    if figures is not None:
        for name, fig in figures.items():
            fig_path = os.path.join(output_dir, f'{name}.png')
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')

def format_float(value, decimal_places=2):
    """
    Format a float value with specified decimal places
    
    Args:
        value: Value to format
        decimal_places (int): Number of decimal places
        
    Returns:
        str: Formatted string
    """
    return f"{value:.{decimal_places}f}"

def format_percentage(value, decimal_places=2):
    """
    Format a value as a percentage with specified decimal places
    
    Args:
        value: Value to format
        decimal_places (int): Number of decimal places
        
    Returns:
        str: Formatted percentage string
    """
    return f"{value:.{decimal_places}f}%"
