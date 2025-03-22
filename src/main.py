"""
Main entry point for the trading entry point prediction project
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from data.loader import load_data, preprocess_data
from data.features import prepare_features
from models.random_forest_model import RandomForestModel
from visualization.charts import (plot_price_chart, plot_entry_points, 
                                 plot_feature_importance, plot_technical_indicators, 
                                 plot_backtest_results)
from backtesting.backtester import backtest_strategy, analyze_backtest_results, generate_backtest_report
from utils.helpers import create_output_directory, save_results

def main(args):
    """
    Run the complete trading model pipeline
    
    Args:
        args: Command line arguments
    """
    print("=== AI Trading Entry Point Prediction ===")
    
    # Set path to data file
    data_file = args.data_file
    
    # Create output directory
    output_dir = create_output_directory()
    print(f"Results will be saved to: {output_dir}")
    
    # Step 1: Load and preprocess data
    print("\nStep 1: Loading and preprocessing data...")
    df_raw = load_data(data_file)
    df = preprocess_data(df_raw)
    
    # Step 2: Prepare features and target
    print("\nStep 2: Preparing features and target...")
    df_features = prepare_features(
        df, 
        include_target=True, 
        future_periods=args.future_periods, 
        profit_target=args.profit_target, 
        stop_loss=args.stop_loss
    )
    
    # Step 3: Split data for training and testing
    print("\nStep 3: Splitting data for training and testing...")
    split_idx = int(len(df_features) * (1 - args.test_size))
    df_train = df_features.iloc[:split_idx]
    df_test = df_features.iloc[split_idx:]
    
    print(f"Training set size: {len(df_train)} rows")
    print(f"Testing set size: {len(df_test)} rows")
    
    # Step 4: Create model instance
    print("\nStep 4: Creating model...")
    model = RandomForestModel(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=args.random_state,
        profit_target=args.profit_target,
        stop_loss=args.stop_loss
    )
    
    # Step 5: Extract features and target
    print("\nStep 5: Extracting features and target...")
    X_train, y_train = model.extract_features_target(df_train)
    X_test, y_test = model.extract_features_target(df_test)
    
    # Step 6: Train model
    print("\nStep 6: Training model...")
    model.train(X_train, y_train)
    
    # Step 7: Evaluate model
    print("\nStep 7: Evaluating model...")
    metrics = model.evaluate(X_test, y_test)
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Step 8: Get feature importance
    print("\nStep 8: Analyzing feature importance...")
    feature_importance = model.get_feature_importance()
    print("\nTop 10 important features:")
    print(feature_importance.head(10))
    
    # Step 9: Find entry points
    print("\nStep 9: Finding entry points...")
    # Get the dates and prices from the test data for entry points
    test_dates = df_test.index
    test_prices = df_test['Close'].values
    
    entry_points = model.find_entry_points(
        X_test, 
        test_dates, 
        test_prices, 
        confidence_threshold=args.confidence_threshold
    )
    
    print(f"Found {len(entry_points)} entry points with confidence >= {args.confidence_threshold}")
    
    # Step 10: Backtest the strategy
    print("\nStep 10: Backtesting strategy...")
    backtest_results = backtest_strategy(
        df_test, 
        entry_points, 
        n_future_bars=args.future_periods
    )
    
    # Step 11: Analyze backtest results
    print("\nStep 11: Analyzing backtest results...")
    backtest_analysis = analyze_backtest_results(backtest_results)
    backtest_report = generate_backtest_report(backtest_analysis)
    print(backtest_report)
    
    # Step 12: Generate visualizations
    print("\nStep 12: Generating visualizations...")
    figures = {}
    
    # Price chart
    figures['price_chart'] = plot_price_chart(df_test, days=60)
    
    # Entry points chart
    figures['entry_points'] = plot_entry_points(df_test, entry_points, days=60)
    
    # Feature importance chart
    figures['feature_importance'] = plot_feature_importance(feature_importance, top_n=15)
    
    # Technical indicators chart
    figures['technical_indicators'] = plot_technical_indicators(df_test, days=30)
    
    # Backtest results chart
    figures['backtest_results'] = plot_backtest_results(backtest_results)
    
    # Step 13: Save results
    print(f"\nStep 13: Saving all results to {output_dir}...")
    save_results(
        output_dir,
        model=model,
        feature_importance=feature_importance,
        entry_points=entry_points,
        backtest_results=backtest_results,
        backtest_analysis=backtest_analysis,
        figures=figures
    )
    
    # Write backtest report to file
    with open(os.path.join(output_dir, 'backtest_report.txt'), 'w') as f:
        f.write(backtest_report)
    
    print("\nDone! All results saved.")

def get_args():
    """Parse command line arguments"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run trading entry point prediction model')
    
    # Data file
    parser.add_argument('--data_file', type=str,
                        default='USATECH.IDXUSD_Candlestick_15_M_BID_01.01.2023-18.01.2025.csv',
                        help='Path to input data file')
    
    # Model parameters
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='Number of trees in the random forest')
    parser.add_argument('--max_depth', type=int, default=10,
                        help='Maximum depth of trees')
    parser.add_argument('--min_samples_split', type=int, default=10,
                        help='Minimum samples required to split a node')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Trading parameters
    parser.add_argument('--profit_target', type=float, default=0.01,
                        help='Profit target as decimal (e.g., 0.01 = 1%)')
    parser.add_argument('--stop_loss', type=float, default=0.005,
                        help='Stop loss as decimal (e.g., 0.005 = 0.5%)')
    parser.add_argument('--future_periods', type=int, default=10,
                        help='Number of periods to look ahead for target')
    
    # Training parameters
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--confidence_threshold', type=float, default=0.6,
                        help='Minimum confidence threshold for entry points')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)
