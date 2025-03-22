"""
Data loading and preprocessing functions
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_data(file_path):
    """
    Load OHLCV data from CSV file.
    Handles European date format (DD.MM.YYYY) in 'Local time' column.
    
    Args:
        file_path (str): Path to CSV file
        
    Returns:
        pd.DataFrame: DataFrame with datetime index
    """
    try:
        # First try using pandas date parsing with European format
        df = pd.read_csv(file_path, parse_dates=['Local time'], dayfirst=True)
        
        # Check if conversion worked correctly
        if not pd.api.types.is_datetime64_any_dtype(df['Local time']):
            raise ValueError("Date parsing failed, trying custom parser")
    except Exception:
        # If standard parsing fails, use custom parsing
        df = pd.read_csv(file_path)
        df['Local time'] = df['Local time'].apply(_parse_datetime)
    
    # Return DataFrame with proper column names
    return df

def _parse_datetime(dt_str):
    """
    Custom datetime parsing function for complex date formats.
    
    Args:
        dt_str (str): Datetime string to parse
        
    Returns:
        datetime: Parsed datetime object
    """
    try:
        # Split by space to separate date, time and timezone
        parts = dt_str.split(' ')
        
        # Extract date parts (DD.MM.YYYY)
        date_parts = parts[0].split('.')
        if len(date_parts) == 3:
            # Rearrange to YYYY-MM-DD format
            date_str = f"{date_parts[2]}-{date_parts[1]}-{date_parts[0]}"
            
            # Extract time part (without timezone)
            time_str = parts[1]
            
            # Combine date and time, ignore timezone for now
            combined = f"{date_str} {time_str}"
            return pd.to_datetime(combined)
        
        # Fallback if date format is different
        return pd.to_datetime(dt_str, dayfirst=True)
    except Exception as e:
        print(f"Error parsing date: {dt_str}, Error: {e}")
        # Last resort, try with errors='coerce'
        return pd.to_datetime(dt_str, errors='coerce', dayfirst=True)

def preprocess_data(df):
    """
    Preprocess raw DataFrame:
    - Rename columns
    - Set timestamp as index
    - Sort by timestamp
    
    Args:
        df (pd.DataFrame): Raw DataFrame
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Rename columns if needed
    if 'Local time' in df.columns:
        df.rename(columns={'Local time': 'timestamp'}, inplace=True)
    
    # Set timestamp as index
    if 'timestamp' in df.columns:
        df.set_index('timestamp', inplace=True)
    
    # Sort by timestamp
    df.sort_index(inplace=True)
    
    return df

def resample_data(df, timeframe):
    """
    Resample OHLCV data to a different timeframe.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        timeframe (str): Pandas resampling rule (e.g., '1H', '4H', '1D')
        
    Returns:
        pd.DataFrame: Resampled DataFrame
    """
    # Validate input data
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain OHLCV columns: {required_cols}")
    
    # Define resampling functions
    resampled = df.resample(timeframe).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    
    # Drop any NaN rows that might result from resampling
    resampled.dropna(inplace=True)
    
    return resampled
