"""
Tests for the data loader module
"""

import os
import pandas as pd
import pytest
from src.data.loader import load_data, preprocess_data

# Define test data file paths
test_data_path = os.path.join(os.path.dirname(__file__), 'test_data')
os.makedirs(test_data_path, exist_ok=True)
test_csv_path = os.path.join(test_data_path, 'test_ohlcv.csv')

# Create test data if it doesn't exist
if not os.path.exists(test_csv_path):
    # Create test data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='15min')
    data = {
        'Local time': [d.strftime('%d.%m.%Y %H:%M:%S.000') for d in dates],
        'Open': [100 + i * 0.1 for i in range(100)],
        'High': [102 + i * 0.1 for i in range(100)],
        'Low': [99 + i * 0.1 for i in range(100)],
        'Close': [101 + i * 0.1 for i in range(100)],
        'Volume': [1000 + i * 10 for i in range(100)]
    }
    df = pd.DataFrame(data)
    df.to_csv(test_csv_path, index=False)

def test_load_data():
    """Test loading data from CSV file"""
    df = load_data(test_csv_path)
    
    # Check that the dataframe has expected columns
    assert 'Local time' in df.columns
    assert 'Open' in df.columns
    assert 'High' in df.columns
    assert 'Low' in df.columns
    assert 'Close' in df.columns
    assert 'Volume' in df.columns
    
    # Check that dates are parsed correctly
    assert pd.api.types.is_datetime64_any_dtype(df['Local time'])
    
    # Check that numeric columns have correct types
    assert pd.api.types.is_numeric_dtype(df['Open'])
    assert pd.api.types.is_numeric_dtype(df['High'])
    assert pd.api.types.is_numeric_dtype(df['Low'])
    assert pd.api.types.is_numeric_dtype(df['Close'])
    assert pd.api.types.is_numeric_dtype(df['Volume'])
    
    # Check that we have the expected number of rows
    assert len(df) == 100

def test_preprocess_data():
    """Test preprocessing data"""
    # Load data
    df_raw = load_data(test_csv_path)
    
    # Preprocess data
    df = preprocess_data(df_raw)
    
    # Check that timestamp is now the index
    assert df.index.name == 'timestamp'
    
    # Check that the dataframe has expected columns
    assert 'Open' in df.columns
    assert 'High' in df.columns
    assert 'Low' in df.columns
    assert 'Close' in df.columns
    assert 'Volume' in df.columns
    
    # Check that the index is sorted
    assert df.index.is_monotonic_increasing
    
    # Check that we have the expected number of rows
    assert len(df) == 100
