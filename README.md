# TradeSight-AI :: AI Trading Entry Point Prediction Project

## Title
TradeSight-AI: Intelligent Trading Entry Point Prediction

## Short Description
AI-powered trading assistant that identifies optimal market entry points with explainable rationale, technical analysis, and visual insights.

## Detailed Description
TradeSight-AI leverages machine learning to identify high-probability trading opportunities in financial markets. This project combines advanced technical analysis, machine learning, and explainable AI to provide traders with actionable insights.

**Key Features:**
- 🎯 Precise entry point detection using Random Forest models
- 📊 Comprehensive technical indicator analysis (RSI, MACD, Bollinger Bands, etc.)
- 📈 Interactive visualizations of predictions and backtests
- 💡 Natural language explanations for every trading signal
- 📝 Detailed backtesting framework with performance metrics
- 🔄 Modular architecture for easy customization and extension

Built with Python, scikit-learn, and Plotly, TradeSight-AI helps traders make more informed decisions by identifying patterns that human analysis might miss while providing clear reasoning behind each recommendation.

## Tags/Topics
`trading`, `machine-learning`, `financial-analysis`, `technical-indicators`, `python`, `data-science`, `algorithmic-trading`, `ai`, `explainable-ai`, `visualization`

This project implements an AI model for determining optimal trading entry points with supporting rationale. The model uses machine learning to identify potential trading opportunities and provides clear explanations for its recommendations.

## Project Structure

```
├── data/                    # CSV data files (USATECH index)
├── notebooks/               # Jupyter notebooks for exploration and analysis
│   ├── 01_data_exploration.ipynb     # Basic data exploration
│   ├── 02_feature_engineering.ipynb  # Feature engineering and preparation
│   ├── 03_model_training.ipynb       # Model training and evaluation
│   ├── 04_backtesting.ipynb          # Strategy backtesting
│   └── 05_visualization.ipynb        # Advanced visualizations
├── results/                 # Trading results and model outputs
├── src/                     # Source code organized by module
│   ├── data/                # Data loading and feature engineering
│   ├── models/              # Model implementation
│   ├── visualization/       # Visualization utilities
│   ├── backtesting/         # Backtesting functionality
│   └── utils/               # Shared utilities
├── main.py                  # Main CLI entry point
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Features

- **Data Preparation**: Load, preprocess, and feature engineering for trading data
- **Advanced Feature Engineering**: Technical indicators, price patterns, volatility metrics
- **Model Training**: Random Forest model with trained weights
- **Backtesting**: Test the model against historical data
- **Visualization**: Interactive charts and dashboards
- **Explainability**: Natural language explanations for trading signals

## Prerequisites

- Python 3.8+
- Required packages in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd trading
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Interactive Notebooks

The easiest way to explore the project is through the Jupyter notebooks:

```bash
jupyter notebook notebooks/
```

The notebooks are organized in sequence:
1. Data exploration
2. Feature engineering
3. Model training
4. Backtesting
5. Visualization

### Command Line Interface

You can also run the model from the command line:

```bash
python src/main.py --data_file USATECH.IDXUSD_Candlestick_15_M_BID_01.01.2023-18.01.2025.csv
```

Optional arguments:
- `--confidence_threshold`: Minimum confidence for entry points (default: 0.6)
- `--profit_target`: Target profit percentage (default: 0.01)
- `--stop_loss`: Stop loss percentage (default: 0.005)
- `--future_periods`: Periods to look ahead for target (default: 10)

For more options:
```bash
python src/main.py --help
```

## Data Sources

The project uses USATECH index candlestick data:
- 15-minute timeframe: `USATECH.IDXUSD_Candlestick_15_M_BID_01.01.2023-18.01.2025.csv`
- 5-minute timeframe: `USATECH.IDXUSD_Candlestick_5_M_BID_01.01.2023-18.01.2025.csv`

## Model Approach

Our entry point prediction model uses a Random Forest classifier trained on historical price data with various technical indicators. The model looks for patterns that precede profitable trade opportunities based on a defined risk-reward ratio.

Key features include:
- Moving averages (SMA, EMA)
- Oscillators (RSI, MACD)
- Volatility measures (Bollinger Bands, ATR)
- Price patterns and crossover signals
- Volume indicators

## Results and Visualization

The model generates several visualizations to help understand its predictions:
- Price charts with entry points
- Technical indicator dashboards
- Trade result analysis
- Performance metrics
- Feature importance analysis

Results are saved in the `results/` directory with timestamped folders for each run.

## Customization

The model is designed to be flexible and can be customized in several ways:
- Change the target asset by providing different data files
- Adjust profit target and stop loss parameters
- Modify the feature engineering process
- Implement different model architectures

## License

Use of this project is governed by the MIT License found in the [LICENSE](LICENSE) file.

## Acknowledgments

We would like to thank all contributors and resources that made this project possible. Special thanks to the open-source community for their invaluable tools and libraries.
