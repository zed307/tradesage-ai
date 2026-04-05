
# TradeSage AI – Unrestricted Trader's Edge

A professional-grade Streamlit application for quantitative trading research, backtesting, and machine learning-powered strategy development. Combines live market data, sentiment analysis, technical indicators, and advanced ML models (XGBoost, LightGBM, LSTM, ensemble methods) for comprehensive strategy evaluation.

## Features

### Core Tabs
- **Overview**: Real-time market data, key metrics, portfolio summary
- **Live Chart + Indicators**: Interactive candlestick charts with technical indicators (RSI, MACD, Bollinger Bands, EMA, SMA, ATR, etc.)
- **News & Events**: Curated financial news and market-moving events
- **Sentiment Analysis**: Real-time sentiment scoring from news feeds
- **Trader Predictions**: AI-generated daily trading signals and recommendations
- **Backtester**: Classical strategies + ML-powered backtesting with walk-forward validation

### Backtester – Classical Strategies
- SMA/EMA Crossover
- RSI Oversold/Overbought
- MACD Signal Crossover
- Bollinger Bands Mean Reversion
- ATR-based Volatility Strategies

### Backtester – Machine Learning Strategies
- **XGBoost / LightGBM**: Tabular feature-based return/direction prediction
- **Random Forest**: Robust ensemble baseline
- **Logistic Regression / SVM**: Simple directional signals
- **LSTM / GRU**: Temporal sequence modeling
- **Hybrid LSTM + XGBoost**: Temporal features + tree-based final prediction
- **Ensemble Voting/Stacking**: Combined model predictions

### ML Features
- **Automated Feature Engineering**: Lagged returns, technical indicators, sentiment scores, time-based features, regime detection
- **Walk-Forward Optimization**: Time-series aware rolling train/test windows (no future leakage)
- **Hyperparameter Tuning**: Grid search, random search, Optuna integration
- **Realistic Execution**: Slippage, commission, spread, position sizing
- **Comprehensive Metrics**: CAGR, Sharpe, Sortino, Calmar, Max DD, Win Rate, Profit Factor
- **Visualizations**: Equity curves, feature importance, prediction accuracy, confusion matrices, regime analysis
- **Model Explainability**: SHAP values, permutation importance, feature interaction analysis

## Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### Clone & Setup

```bash
# Clone the repository
git clone https://github.com/zed307/tradesage-ai.git
cd tradesage-ai

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

## Configuration

- **API Keys**: (Optional) Add API keys to `.env` for enhanced data sources
- **Symbols**: Change trading pair in the sidebar (BTC/USDT, ETH/USDT, AAPL, etc.)
- **Timeframe**: Select from 1m, 5m, 15m, 1h, 1d, 1w
- **Backtesting Parameters**: Adjust capital, position size, slippage, commission via sidebar

## Data Sources

- **Historical Data**: yfinance (stocks, ETFs, indices), CCXT (cryptocurrency exchanges)
- **News & Sentiment**: RSS feeds, financial APIs
- **Real-time Quotes**: CCXT exchanges, yfinance

## Architecture

- **Frontend**: Streamlit (interactive widgets, real-time updates)
- **Data Processing**: Pandas, NumPy
- **Technical Analysis**: Pandas-TA
- **ML Models**: Scikit-learn, XGBoost, LightGBM, TensorFlow/Keras
- **Visualization**: Plotly (interactive charts)
- **Backtesting**: Vectorized operations with position sizing and realistic costs

## Professional Use

This tool is designed for quant researchers, traders, and financial technologists who need:
- Rapid strategy prototyping and validation
- Feature engineering and ML model experimentation
- Out-of-sample performance evaluation
- Professional-grade backtesting with realistic execution assumptions
- Multi-strategy comparison and regime analysis

## ML Backtesting Workflow

1. **Select Trading Pair & Timeframe** (sidebar)
2. **Choose ML Model** (XGBoost, LightGBM, LSTM, etc.)
3. **Configure Feature Engineering** (toggle indicators, sentiment, trader predictions)
4. **Tune Hyperparameters** (learning rate, tree depth, epochs, etc.)
5. **Train & Backtest** – Walk-forward validation prevents overfitting
6. **Analyze Results** – Equity curves, metrics, feature importance, trade logs
7. **Export Report** – JSON/CSV for further analysis

## Experimental Features

- **Sentiment Integration**: News sentiment as ML feature
- **Trader Prediction Fusion**: Daily predictions as ML input signal
- **Regime Detection**: Automatic detection of trending vs mean-reverting periods
- **Multi-Model Comparison**: Compare classical vs ML vs Buy & Hold strategies
- **SHAP Explainability**: (Optional) Model decision explanations

## System Requirements

- **RAM**: 4GB minimum (8GB+ recommended for deep learning)
- **CPU**: Multi-core processor
- **GPU**: Optional (TensorFlow can use NVIDIA GPUs if available)
- **Storage**: ~500MB for app + data cache

## Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt --upgrade
```

### Slow performance
- Reduce lookback period in sidebar
- Use simpler ML models (Random Forest instead of LSTM)
- Disable SHAP explanations for faster training

### Data loading issues
- Check internet connection
- Verify symbol exists (e.g., BTC/USDT for crypto, AAPL for stocks)
- Try different timeframe

### TensorFlow/GPU issues
- For CPU-only mode, TensorFlow works out-of-box
- For GPU: Install CUDA 12.x and cuDNN 8.x, update TensorFlow
- See [TensorFlow GPU setup](https://www.tensorflow.org/install/gpu)

## Disclaimer

This software is for educational and research purposes only. Backtesting results do not guarantee future performance. Use at your own risk. Always conduct due diligence and consult with financial professionals before deploying live trading strategies.

**Key Limitations:**
- Past performance ≠ future results
- Backtests ignore slippage, market impact, and execution gaps
- ML models can overfit to historical data
- Real trading involves psychology, regulatory, and operational risks not modeled here

## License

MIT License – See LICENSE file for details

## Contributing

Contributions welcome! Fork, branch, and submit PRs for:
- New ML models or indicators
- Bug fixes
- Documentation improvements
- Performance optimizations

## Contact & Support

- **Issues**: Open GitHub issues for bugs/feature requests
- **Discussions**: Use GitHub Discussions for ideas and questions
- **Email**: (Contact info if provided)

---

**Happy quant trading! 📈**
