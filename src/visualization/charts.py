"""
Visualization functions for trading data and model results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_price_chart(df, days=60):
    """
    Plot basic price chart with moving averages
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV and indicator data
        days (int): Number of days to display
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Get recent data for plotting
    if len(df) <= days:
        plot_df = df
    else:
        plot_df = df.iloc[-days:]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot price and moving averages
    ax.plot(plot_df.index, plot_df['Close'], label='Close Price')
    
    # Add moving averages if available
    for ma in ['SMA20', 'SMA50', 'SMA200']:
        if ma in plot_df.columns:
            ax.plot(plot_df.index, plot_df[ma], alpha=0.7, label=ma)
    
    # Format the plot
    ax.set_title(f'Price Chart - Last {days} Days')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format x-axis dates
    plt.xticks(rotation=45)
    
    # Ensure clean layout
    plt.tight_layout()
    
    return fig

def plot_entry_points(df, entry_points, days=60):
    """
    Plot price chart with entry points
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV and indicator data
        entry_points (list): List of entry point dictionaries
        days (int): Number of days to display
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Get recent data for plotting
    if len(df) <= days:
        plot_df = df
    else:
        plot_df = df.iloc[-days:]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot price and moving averages
    ax.plot(plot_df.index, plot_df['Close'], label='Close Price')
    
    # Add moving averages if available
    for ma in ['SMA20', 'SMA50']:
        if ma in plot_df.columns:
            ax.plot(plot_df.index, plot_df[ma], alpha=0.7, label=ma)
    
    # Plot entry points
    for entry in entry_points:
        # Skip if entry date is not in the plot range
        if entry['date'] not in plot_df.index:
            continue
            
        # Determine marker based on direction
        color = 'green' if entry['direction'] == 'LONG' else 'red'
        marker = '^' if entry['direction'] == 'LONG' else 'v'
        
        # Plot entry point
        ax.scatter(entry['date'], entry['price'], color=color, marker=marker, s=100,
                   label=f"{entry['direction']} Entry" if entry['direction'] not in [e['direction'] for e in entry_points[:entry_points.index(entry)]] else "")
        
        # Plot target and stop loss
        ax.plot([entry['date'], entry['date']], 
                [entry['price'], entry['target_price']], 
                color=color, linestyle='--', alpha=0.5)
        ax.plot([entry['date'], entry['date']], 
                [entry['price'], entry['stop_loss_price']], 
                color='black', linestyle='--', alpha=0.5)
    
    # Format the plot
    ax.set_title(f'Price Chart with Entry Points - Last {days} Days')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid(True, alpha=0.3)
    
    # Handle legend duplication
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    # Format x-axis dates
    plt.xticks(rotation=45)
    
    # Ensure clean layout
    plt.tight_layout()
    
    return fig

def plot_feature_importance(feature_importance, top_n=15):
    """
    Plot feature importance
    
    Args:
        feature_importance (pd.DataFrame): Feature importance DataFrame with 'Feature' and 'Importance' columns
        top_n (int): Number of top features to display
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Take top N features
    plot_df = feature_importance.head(top_n)
    
    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(plot_df['Feature'], plot_df['Importance'], color='skyblue')
    
    # Add values at the end of bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.002, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center')
    
    # Format the plot
    ax.set_title(f'Top {top_n} Feature Importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.invert_yaxis()  # Highest importance at the top
    
    # Ensure clean layout
    plt.tight_layout()
    
    return fig

def plot_technical_indicators(df, days=60):
    """
    Plot multiple technical indicators
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV and indicator data
        days (int): Number of days to display
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Get recent data for plotting
    if len(df) <= days:
        plot_df = df
    else:
        plot_df = df.iloc[-days:]
    
    # Create plot with subplots
    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    
    # Plot 1: Price and moving averages
    axs[0].plot(plot_df.index, plot_df['Close'], label='Close')
    
    # Add moving averages if available
    for ma in ['SMA20', 'SMA50', 'SMA200']:
        if ma in plot_df.columns:
            axs[0].plot(plot_df.index, plot_df[ma], alpha=0.7, label=ma)
    
    # Add Bollinger Bands if available
    if all(col in plot_df.columns for col in ['BB_Upper', 'BB_Lower']):
        axs[0].fill_between(plot_df.index, plot_df['BB_Upper'], plot_df['BB_Lower'], 
                            color='gray', alpha=0.2, label='Bollinger Bands')
    
    axs[0].set_title('Price and Moving Averages')
    axs[0].set_ylabel('Price')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc='upper left')
    
    # Plot 2: MACD
    if all(col in plot_df.columns for col in ['MACD', 'MACD_Signal']):
        axs[1].plot(plot_df.index, plot_df['MACD'], label='MACD')
        axs[1].plot(plot_df.index, plot_df['MACD_Signal'], label='Signal')
        
        # Add histogram if available
        if 'MACD_Hist' in plot_df.columns:
            # Use bar plot for histogram with positive/negative colors
            colors = ['green' if val >= 0 else 'red' for val in plot_df['MACD_Hist']]
            axs[1].bar(plot_df.index, plot_df['MACD_Hist'], color=colors, alpha=0.5, label='Histogram')
        
        axs[1].set_title('MACD')
        axs[1].set_ylabel('Value')
        axs[1].grid(True, alpha=0.3)
        axs[1].legend(loc='upper left')
        axs[1].axhline(y=0, color='black', linestyle='-', alpha=0.2)
    
    # Plot 3: RSI
    if 'RSI' in plot_df.columns:
        axs[2].plot(plot_df.index, plot_df['RSI'], label='RSI', color='purple')
        axs[2].axhline(y=70, color='red', linestyle='--', alpha=0.5)
        axs[2].axhline(y=30, color='green', linestyle='--', alpha=0.5)
        axs[2].set_title('RSI')
        axs[2].set_ylabel('Value')
        axs[2].set_ylim(0, 100)
        axs[2].grid(True, alpha=0.3)
        axs[2].legend(loc='upper left')
    
    # Plot 4: Volume
    axs[3].bar(plot_df.index, plot_df['Volume'], label='Volume', alpha=0.7, color='blue')
    
    # Add volume moving average if available
    if 'Volume_SMA20' in plot_df.columns:
        axs[3].plot(plot_df.index, plot_df['Volume_SMA20'], color='orange', label='Volume MA')
    
    axs[3].set_title('Volume')
    axs[3].set_ylabel('Volume')
    axs[3].set_xlabel('Date')
    axs[3].grid(True, alpha=0.3)
    axs[3].legend(loc='upper left')
    
    # Format x-axis dates
    plt.xticks(rotation=45)
    
    # Ensure clean layout
    plt.tight_layout()
    
    return fig

def plot_backtest_results(results):
    """
    Plot backtest results
    
    Args:
        results (list): List of trade result dictionaries
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Convert to DataFrame for easier manipulation
    df_results = pd.DataFrame(results)
    
    # Create plot with subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
    
    # Plot 1: Profit/Loss per trade
    if 'profit_pct' in df_results.columns:
        colors = ['green' if x >= 0 else 'red' for x in df_results['profit_pct']]
        axs[0].bar(range(len(df_results)), df_results['profit_pct'], color=colors)
        axs[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axs[0].set_title('Profit/Loss per Trade (%)')
        axs[0].set_ylabel('Profit/Loss (%)')
        axs[0].set_xlabel('Trade #')
        axs[0].grid(True, alpha=0.3)
    
    # Plot 2: Cumulative profit
    if 'profit_pct' in df_results.columns:
        cumulative = df_results['profit_pct'].cumsum()
        axs[1].plot(range(len(cumulative)), cumulative, marker='o')
        axs[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axs[1].set_title('Cumulative Profit (%)')
        axs[1].set_ylabel('Cumulative Profit (%)')
        axs[1].set_xlabel('Trade #')
        axs[1].grid(True, alpha=0.3)
    
    # Plot 3: Outcome distribution
    if 'outcome' in df_results.columns:
        outcomes = df_results['outcome'].value_counts()
        colors = {'WIN': 'green', 'LOSS': 'red', 'INCOMPLETE': 'gray'}
        outcome_colors = [colors.get(outcome, 'blue') for outcome in outcomes.index]
        
        axs[2].bar(outcomes.index, outcomes.values, color=outcome_colors)
        axs[2].set_title('Trade Outcomes')
        axs[2].set_ylabel('Count')
        axs[2].grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        total = outcomes.sum()
        for i, (outcome, count) in enumerate(outcomes.items()):
            percentage = count / total * 100
            axs[2].text(i, count + 0.5, f'{percentage:.1f}%', 
                        ha='center', va='bottom')
    
    # Ensure clean layout
    plt.tight_layout()
    
    return fig
