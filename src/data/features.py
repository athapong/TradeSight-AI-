"""
Feature engineering for trading data
"""

import pandas as pd
import numpy as np

# Try to import talib, fall back to alternative if not available
try:
    import talib
    USING_TALIB = True
except ImportError:
    USING_TALIB = False
    
    # Define fallback functions for technical indicators
    # These are simple implementations and may differ slightly from TA-Lib
    def SMA(series, timeperiod):
        """Simple Moving Average fallback"""
        return series.rolling(window=timeperiod).mean()
        
    def EMA(series, timeperiod):
        """Exponential Moving Average fallback"""
        return series.ewm(span=timeperiod, adjust=False).mean()
        
    def RSI(series, timeperiod):
        """Relative Strength Index fallback"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=timeperiod).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=timeperiod).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def MACD(series, fastperiod, slowperiod, signalperiod):
        """Moving Average Convergence Divergence fallback"""
        fast_ema = EMA(series, fastperiod)
        slow_ema = EMA(series, slowperiod)
        macd_line = fast_ema - slow_ema
        signal_line = EMA(macd_line, signalperiod)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
        
    def ATR(high, low, close, timeperiod):
        """Average True Range fallback"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        return tr.rolling(window=timeperiod).mean()
        
    def BBANDS(series, timeperiod, nbdevup, nbdevdn):
        """Bollinger Bands fallback"""
        sma = SMA(series, timeperiod)
        std = series.rolling(window=timeperiod).std()
        upper = sma + nbdevup * std
        lower = sma - nbdevdn * std
        return upper, sma, lower
        
    def ADX(high, low, close, timeperiod):
        """Average Directional Index fallback"""
        # Simplified ADX calculation
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm.abs()), 0)
        minus_dm = minus_dm.abs().where((minus_dm > 0) & (minus_dm > plus_dm), 0)
        tr = ATR(high, low, close, 1)  # TR for one period
        plus_di = 100 * (EMA(plus_dm, timeperiod) / EMA(tr, timeperiod))
        minus_di = 100 * (EMA(minus_dm, timeperiod) / EMA(tr, timeperiod))
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = EMA(dx, timeperiod)
        return adx

def add_technical_indicators(df):
    """
    Add technical indicators to OHLCV DataFrame
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Use either TA-Lib or fallback functions based on availability
    if USING_TALIB:
        # Moving averages
        df['SMA20'] = talib.SMA(df['Close'], timeperiod=20)
        df['SMA50'] = talib.SMA(df['Close'], timeperiod=50)
        df['SMA200'] = talib.SMA(df['Close'], timeperiod=200)
        df['EMA12'] = talib.EMA(df['Close'], timeperiod=12)
        df['EMA26'] = talib.EMA(df['Close'], timeperiod=26)
        
        # Momentum indicators
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Hist'] = macd_hist
        
        # Volatility indicators
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        upper, middle, lower = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        df['BB_Upper'] = upper
        df['BB_Middle'] = middle
        df['BB_Lower'] = lower
        
        # Trend indicators
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    else:
        # Moving averages
        df['SMA20'] = SMA(df['Close'], timeperiod=20)
        df['SMA50'] = SMA(df['Close'], timeperiod=50)
        df['SMA200'] = SMA(df['Close'], timeperiod=200)
        df['EMA12'] = EMA(df['Close'], timeperiod=12)
        df['EMA26'] = EMA(df['Close'], timeperiod=26)
        
        # Momentum indicators
        df['RSI'] = RSI(df['Close'], timeperiod=14)
        macd, macd_signal, macd_hist = MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Hist'] = macd_hist
        
        # Volatility indicators
        df['ATR'] = ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        upper, middle, lower = BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        df['BB_Upper'] = upper
        df['BB_Middle'] = middle
        df['BB_Lower'] = lower
        
        # Trend indicators
        df['ADX'] = ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    
    return df

def add_price_features(df):
    """
    Add price-based features to DataFrame with technical indicators
    
    Args:
        df (pd.DataFrame): DataFrame with technical indicators
        
    Returns:
        pd.DataFrame: DataFrame with additional price features
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Price position relative to indicators
    df['Close_SMA20_Ratio'] = df['Close'] / df['SMA20'].replace(0, np.nan)
    df['Close_SMA50_Ratio'] = df['Close'] / df['SMA50'].replace(0, np.nan)
    df['Close_SMA200_Ratio'] = df['Close'] / df['SMA200'].replace(0, np.nan)
    
    # BB position
    bb_width = (df['BB_Upper'] - df['BB_Lower']).replace(0, np.nan)
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / bb_width
    
    # Crossover signals
    df['Golden_Cross'] = ((df['SMA50'].shift(1) < df['SMA200'].shift(1)) & 
                         (df['SMA50'] >= df['SMA200'])).astype(int)
    df['Death_Cross'] = ((df['SMA50'].shift(1) > df['SMA200'].shift(1)) & 
                        (df['SMA50'] <= df['SMA200'])).astype(int)
    
    # MACD crossover
    df['MACD_Cross_Above'] = ((df['MACD'].shift(1) < df['MACD_Signal'].shift(1)) & 
                             (df['MACD'] >= df['MACD_Signal'])).astype(int)
    df['MACD_Cross_Below'] = ((df['MACD'].shift(1) > df['MACD_Signal'].shift(1)) & 
                             (df['MACD'] <= df['MACD_Signal'])).astype(int)
    
    # Price change features
    df['Daily_Return'] = df['Close'].pct_change()
    df['Weekly_Return'] = df['Close'].pct_change(5)  # Assuming 5 trading days in a week
    df['Monthly_Return'] = df['Close'].pct_change(20)  # Assuming 20 trading days in a month
    
    # Time-based features
    df['DayOfWeek'] = df.index.dayofweek
    df['HourOfDay'] = df.index.hour
    
    return df

def add_volume_features(df):
    """
    Add volume-based features to DataFrame
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        
    Returns:
        pd.DataFrame: DataFrame with volume features
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Volume moving average
    if USING_TALIB:
        df['Volume_SMA20'] = talib.SMA(df['Volume'], timeperiod=20)
    else:
        df['Volume_SMA20'] = SMA(df['Volume'], timeperiod=20)
    
    # Volume ratio (current volume relative to moving average)
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA20'].replace(0, np.nan)
    
    # High volume days (1.5x average)
    df['High_Volume'] = (df['Volume_Ratio'] > 1.5).astype(int)
    
    # Volume trend (20-day slope of volume)
    df['Volume_Trend'] = df['Volume'].rolling(window=20).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=False
    )
    
    return df

def generate_target(df, future_periods=10, profit_target=0.01, stop_loss=0.005):
    """
    Generate target variable for supervised learning based on profit target and stop loss
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        future_periods (int): Number of periods to look ahead
        profit_target (float): Target profit as a decimal (e.g., 0.01 = 1%)
        stop_loss (float): Stop loss as a decimal (e.g., 0.005 = 0.5%)
    
    Returns:
        pd.DataFrame: DataFrame with added 'Target' column (1=long, -1=short, 0=neutral)
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Initialize target column
    df['Target'] = 0
    
    # Look ahead for each bar to see if we would hit profit target or stop loss
    for i in range(len(df) - future_periods):
        current_price = df['Close'].iloc[i]
        future_slice = df.iloc[i+1:i+future_periods+1]
        
        # Check for long trade (bullish)
        future_high = future_slice['High'].max()
        future_low = future_slice['Low'].min()
        
        # Long trade - hit profit target before stop loss
        if (future_high >= current_price * (1 + profit_target)) and \
           (future_low > current_price * (1 - stop_loss)):
            df.iloc[i, df.columns.get_loc('Target')] = 1
        
        # Short trade - hit profit target before stop loss
        elif (future_low <= current_price * (1 - profit_target)) and \
             (future_high < current_price * (1 + stop_loss)):
            df.iloc[i, df.columns.get_loc('Target')] = -1
    
    return df

def prepare_features(df, include_target=True, future_periods=10, profit_target=0.01, stop_loss=0.005):
    """
    Complete feature preparation pipeline
    
    Args:
        df (pd.DataFrame): Raw OHLCV DataFrame
        include_target (bool): Whether to generate target variable
        future_periods (int): Number of periods to look ahead for target
        profit_target (float): Target profit as a decimal
        stop_loss (float): Stop loss as a decimal
    
    Returns:
        pd.DataFrame: DataFrame with all features and target
    """
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Add price features
    df = add_price_features(df)
    
    # Add volume features
    df = add_volume_features(df)
    
    # Generate target variable if requested
    if include_target:
        df = generate_target(df, future_periods, profit_target, stop_loss)
    
    # Clean up NaN values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    return df
