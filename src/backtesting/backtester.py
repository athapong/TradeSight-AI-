"""
Backtesting functionality for trading strategies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def backtest_strategy(df, entry_points, n_future_bars=10):
    """
    Backtest a trading strategy based on entry points
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        entry_points (list): List of entry point dictionaries
        n_future_bars (int): Number of bars to look forward for determining outcomes
        
    Returns:
        list: List of trade results
    """
    results = []
    
    for entry in entry_points:
        # Find index of entry timestamp
        try:
            # Handle both timestamp as index and as column
            if 'timestamp' in df.columns:
                df_idx = df[df['timestamp'] == entry['date']].index[0]
            else:
                df_idx = df.index.get_loc(entry['date'])
        except:
            # If timestamp not in index, skip this entry
            continue
        
        # Get future data
        future_slice = df.iloc[df_idx+1:df_idx+n_future_bars+1]
        
        # Initialize result
        result = {
            'timestamp': entry['date'],
            'entry_price': entry['price'],
            'direction': entry['direction'],
            'target_price': entry['target_price'],
            'stop_loss_price': entry['stop_loss_price'],
            'outcome': 'UNKNOWN',
            'exit_price': None,
            'exit_timestamp': None,
            'profit_pct': 0,
            'bars_held': 0
        }
        
        # Skip if no future data
        if len(future_slice) == 0:
            result['outcome'] = 'INCOMPLETE'
            results.append(result)
            continue
        
        # Check if target or stop is hit
        for i, (idx, row) in enumerate(future_slice.iterrows()):
            result['bars_held'] = i + 1
            
            if entry['direction'] == 'LONG':
                # Check if target is hit
                if row['High'] >= entry['target_price']:
                    result['outcome'] = 'WIN'
                    result['exit_price'] = entry['target_price']
                    result['exit_timestamp'] = idx
                    result['profit_pct'] = (result['exit_price'] / result['entry_price'] - 1) * 100
                    break
                    
                # Check if stop loss is hit
                if row['Low'] <= entry['stop_loss_price']:
                    result['outcome'] = 'LOSS'
                    result['exit_price'] = entry['stop_loss_price']
                    result['exit_timestamp'] = idx
                    result['profit_pct'] = (result['exit_price'] / result['entry_price'] - 1) * 100
                    break
            else:  # SHORT
                # Check if target is hit
                if row['Low'] <= entry['target_price']:
                    result['outcome'] = 'WIN'
                    result['exit_price'] = entry['target_price']
                    result['exit_timestamp'] = idx
                    result['profit_pct'] = (1 - result['exit_price'] / result['entry_price']) * 100
                    break
                    
                # Check if stop loss is hit
                if row['High'] >= entry['stop_loss_price']:
                    result['outcome'] = 'LOSS'
                    result['exit_price'] = entry['stop_loss_price']
                    result['exit_timestamp'] = idx
                    result['profit_pct'] = (1 - result['exit_price'] / result['entry_price']) * 100
                    break
        
        # If we reach the end without hitting target or stop, use last bar's close
        if result['outcome'] == 'UNKNOWN' and len(future_slice) > 0:
            result['outcome'] = 'INCOMPLETE'
            result['exit_price'] = future_slice.iloc[-1]['Close']
            result['exit_timestamp'] = future_slice.index[-1]
            
            if entry['direction'] == 'LONG':
                result['profit_pct'] = (result['exit_price'] / result['entry_price'] - 1) * 100
            else:
                result['profit_pct'] = (1 - result['exit_price'] / result['entry_price']) * 100
        
        results.append(result)
    
    return results

def analyze_backtest_results(results):
    """
    Analyze backtest results
    
    Args:
        results (list): List of trade result dictionaries
        
    Returns:
        dict: Performance metrics
    """
    if not results:
        # Return an empty analysis dictionary
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'incomplete_trades': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'avg_bars_held': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'total_profit_pct': 0,
            'sharpe_ratio': 0
        }
    
    # Convert to DataFrame for analysis
    df_results = pd.DataFrame(results)
    
    # Calculate basic statistics
    total_trades = len(df_results)
    winning_trades = len(df_results[df_results['outcome'] == 'WIN'])
    losing_trades = len(df_results[df_results['outcome'] == 'LOSS'])
    incomplete_trades = len(df_results[df_results['outcome'] == 'INCOMPLETE'])
    
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    
    avg_profit = df_results['profit_pct'].mean() if 'profit_pct' in df_results else 0
    avg_win = df_results[df_results['profit_pct'] > 0]['profit_pct'].mean() if len(df_results[df_results['profit_pct'] > 0]) > 0 else 0
    avg_loss = df_results[df_results['profit_pct'] < 0]['profit_pct'].mean() if len(df_results[df_results['profit_pct'] < 0]) > 0 else 0
    
    profit_factor = abs(df_results[df_results['profit_pct'] > 0]['profit_pct'].sum() / 
                        df_results[df_results['profit_pct'] < 0]['profit_pct'].sum()) if df_results[df_results['profit_pct'] < 0]['profit_pct'].sum() != 0 else float('inf')
    
    avg_bars_held = df_results['bars_held'].mean() if 'bars_held' in df_results else 0
    
    # Calculate max consecutive wins/losses
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    
    if 'outcome' in df_results:
        current_wins = 0
        current_losses = 0
        
        for outcome in df_results['outcome']:
            if outcome == 'WIN':
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            elif outcome == 'LOSS':
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
    
    # Calculate total profit
    total_profit_pct = df_results['profit_pct'].sum() if 'profit_pct' in df_results else 0
    
    # Calculate Sharpe ratio (simplified)
    returns = df_results['profit_pct'] / 100 if 'profit_pct' in df_results else pd.Series([0])
    sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0
    
    # Combine results
    analysis = {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'incomplete_trades': incomplete_trades,
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'avg_bars_held': avg_bars_held,
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,
        'total_profit_pct': total_profit_pct,
        'sharpe_ratio': sharpe_ratio
    }
    
    return analysis

def generate_backtest_report(analysis):
    """
    Generate a text report from backtest analysis
    
    Args:
        analysis (dict): Backtest analysis results
        
    Returns:
        str: Formatted report text
    """
    report = "=== BACKTEST RESULTS ===\n\n"
    
    # Overall statistics
    report += "OVERALL STATISTICS:\n"
    report += f"Total trades: {analysis['total_trades']}\n"
    report += f"Win rate: {analysis['win_rate']:.2f}%\n"
    report += f"Total profit: {analysis['total_profit_pct']:.2f}%\n"
    report += f"Sharpe ratio: {analysis['sharpe_ratio']:.2f}\n"
    report += f"Profit factor: {analysis['profit_factor']:.2f}\n\n"
    
    # Trade breakdown
    report += "TRADE BREAKDOWN:\n"
    report += f"Winning trades: {analysis['winning_trades']} ({analysis['winning_trades']/analysis['total_trades']*100:.1f}% of total)\n"
    report += f"Losing trades: {analysis['losing_trades']} ({analysis['losing_trades']/analysis['total_trades']*100:.1f}% of total)\n"
    report += f"Incomplete trades: {analysis['incomplete_trades']} ({analysis['incomplete_trades']/analysis['total_trades']*100:.1f}% of total)\n\n"
    
    # Trade metrics
    report += "TRADE METRICS:\n"
    report += f"Average profit per trade: {analysis['avg_profit']:.2f}%\n"
    report += f"Average winning trade: +{analysis['avg_win']:.2f}%\n"
    report += f"Average losing trade: {analysis['avg_loss']:.2f}%\n"
    report += f"Average holding period: {analysis['avg_bars_held']:.1f} bars\n"
    report += f"Max consecutive wins: {analysis['max_consecutive_wins']}\n"
    report += f"Max consecutive losses: {analysis['max_consecutive_losses']}\n"
    
    return report
