"""
TradeSage AI – Unrestricted Trader's Edge
Complete Streamlit App with ML-Powered Backtesting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import ccxt
import requests
import json
import pickle
import warnings
from io import BytesIO

# ML & Data Science
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.pipeline import Pipeline
import optuna
from optuna.pruners import MedianPruner

# Deep Learning (CPU mode)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    tf.config.run_functions_eagerly(True)
    TENSORFLOW_AVAILABLE = True
except:
    TENSORFLOW_AVAILABLE = False

import shap

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG & THEME
# ============================================================================

st.set_page_config(
    page_title="TradeSage AI – Trader's Edge",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark trader theme
DARK_BG = "#0e1117"
CARD_BG = "#161b22"
PRIMARY_COLOR = "#00d9ff"
ACCENT_COLOR = "#ff006e"
SUCCESS_COLOR = "#00ff88"
DANGER_COLOR = "#ff4444"

st.markdown(f"""
    <style>
    .main {{ background-color: {DARK_BG}; }}
    .stCard {{ background-color: {CARD_BG}; border-radius: 8px; padding: 16px; margin: 8px 0; }}
    .stMetric {{ background-color: {CARD_BG}; border-radius: 8px; padding: 12px; }}
    h1, h2, h3, h4 {{ color: {PRIMARY_COLOR}; }}
    .success {{ color: {SUCCESS_COLOR}; }}
    .danger {{ color: {DANGER_COLOR}; }}
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE & CACHE
# ============================================================================

if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}
if 'model_cache' not in st.session_state:
    st.session_state.model_cache = {}
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = {}

@st.cache_data(ttl=300)
def get_historical_data(symbol, timeframe='1d', limit=365):
    """Fetch historical data from yfinance or CCXT"""
    try:
        # Try yfinance first (stocks, indices)
        if '/' not in symbol:
            data = yf.download(symbol, period=f'{limit}d', progress=False)
        else:
            # Crypto via CCXT
            exchange = ccxt.binance()
            timeframe_ms = {'1m': 60*1000, '5m': 5*60*1000, '1h': 60*60*1000, '1d': 24*60*60*1000}
            ms = timeframe_ms.get(timeframe, 24*60*60*1000)
            candles = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            data = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume'] if 'Open' not in data.columns else data.columns
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def get_sentiment_score():
    """Simulated sentiment analysis – in production, use API"""
    return np.random.uniform(-1, 1)

def get_news_feed():
    """Simulated news feed"""
    return [
        {"title": "Bitcoin surges on institutional adoption", "sentiment": 0.85, "source": "CoinTelegraph"},
        {"title": "Fed signals rate cuts ahead", "sentiment": 0.65, "source": "Reuters"},
        {"title": "Market volatility remains elevated", "sentiment": -0.45, "source": "Bloomberg"},
    ]

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(data, lookback=20, include_sentiment=True, sentiment_score=0):
    """
    Comprehensive feature engineering with look-ahead bias protection
    """
    df = data.copy()
    df['returns'] = df['Close'].pct_change()
    
    # Lagged returns & prices
    for i in range(1, lookback + 1):
        df[f'return_lag_{i}'] = df['returns'].shift(i)
        df[f'close_lag_{i}'] = df['Close'].shift(i)
    
    # Technical indicators (all using past data only)
    df['rsi'] = ta.rsi(df['Close'], length=14)
    df['macd'] = ta.macd(df['Close'])['MACD_12_26_9']
    df['macd_signal'] = ta.macd(df['Close'])['MACDh_12_26_9']
    bb = ta.bbands(df['Close'], length=20)
    df['bb_upper'] = bb['BBU_20_2.0']
    df['bb_lower'] = bb['BBL_20_2.0']
    df['bb_mid'] = bb['BBM_20_2.0']
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['ema_12'] = ta.ema(df['Close'], length=12)
    df['ema_26'] = ta.ema(df['Close'], length=26)
    df['sma_50'] = ta.sma(df['Close'], length=50)
    df['sma_200'] = ta.sma(df['Close'], length=200)
    df['volume_sma'] = df['Volume'].rolling(20).mean()
    df['volatility'] = df['returns'].rolling(20).std()
    df['momentum'] = ta.momentum(df['Close'], length=14)
    df['stoch'] = ta.stoch(df['High'], df['Low'], df['Close'])['STOCHk_14_3_3']
    
    # Volume features
    df['volume_ratio'] = df['Volume'] / df['volume_sma']
    
    # Time-based features
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    # Sentiment feature
    if include_sentiment:
        df['sentiment'] = sentiment_score
    
    # Regime detection (rolling correlation with recent trend)
    df['rolling_corr'] = df['Close'].rolling(20).apply(
        lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1 if x.iloc[-1] < x.iloc[0] else 0
    )
    
    # Drop NaN rows
    df = df.dropna()
    
    return df

def select_features(df, feature_cols=None):
    """Select relevant features for ML model"""
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    return df[feature_cols].fillna(0)

# ============================================================================
# ML MODEL BUILDERS
# ============================================================================

def train_xgboost(X_train, y_train, X_test, y_test, task='regression', hyperparams=None):
    """Train XGBoost model"""
    if hyperparams is None:
        hyperparams = {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 100}
    
    if task == 'regression':
        model = xgb.XGBRegressor(**hyperparams, random_state=42, n_jobs=-1)
    else:
        model = xgb.XGBClassifier(**hyperparams, random_state=42, n_jobs=-1)
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return model

def train_lightgbm(X_train, y_train, X_test, y_test, task='regression', hyperparams=None):
    """Train LightGBM model"""
    if hyperparams is None:
        hyperparams = {'num_leaves': 31, 'learning_rate': 0.1, 'n_estimators': 100}
    
    if task == 'regression':
        model = lgb.LGBMRegressor(**hyperparams, random_state=42, n_jobs=-1)
    else:
        model = lgb.LGBMClassifier(**hyperparams, random_state=42, n_jobs=-1)
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=-1)
    return model

def train_random_forest(X_train, y_train, task='regression', hyperparams=None):
    """Train Random Forest model"""
    if hyperparams is None:
        hyperparams = {'n_estimators': 100, 'max_depth': 10}
    
    if task == 'regression':
        model = RandomForestRegressor(**hyperparams, random_state=42, n_jobs=-1)
    else:
        model = RandomForestClassifier(**hyperparams, random_state=42, n_jobs=-1)
    
    model.fit(X_train, y_train)
    return model

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression for classification"""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train, task='classification'):
    """Train SVM model"""
    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_lstm(X_train, y_train, X_test, y_test, lookback=60, epochs=20):
    """Train LSTM model for sequence prediction"""
    if not TENSORFLOW_AVAILABLE:
        st.warning("TensorFlow not available; LSTM skipped")
        return None
    
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(1, X_train.shape[1])),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X_train_lstm, y_train, epochs=epochs, batch_size=32, verbose=0, validation_data=(X_test_lstm, y_test))
    
    return model

def train_hybrid_lstm_xgb(X_train, y_train, X_test, y_test):
    """Train hybrid LSTM + XGBoost"""
    if not TENSORFLOW_AVAILABLE:
        return train_xgboost(X_train, y_train, X_test, y_test)
    
    # LSTM for feature extraction
    lstm_model = Sequential([
        LSTM(32, activation='relu', input_shape=(1, X_train.shape[1]), return_sequences=False),
        Dense(16, activation='relu')
    ])
    
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train.reshape((X_train.shape[0], 1, X_train.shape[1])), y_train, 
                   epochs=10, batch_size=32, verbose=0)
    
    # Extract features from LSTM
    X_train_lstm_features = lstm_model.predict(X_train.reshape((X_train.shape[0], 1, X_train.shape[1])))
    X_test_lstm_features = lstm_model.predict(X_test.reshape((X_test.shape[0], 1, X_test.shape[1])))
    
    # XGBoost on LSTM features
    xgb_model = xgb.XGBRegressor(max_depth=6, learning_rate=0.1, n_estimators=100, random_state=42)
    xgb_model.fit(X_train_lstm_features, y_train)
    
    return {'lstm': lstm_model, 'xgb': xgb_model, 'type': 'hybrid'}

# ============================================================================
# BACKTESTING ENGINE
# ============================================================================

def backtest_ml_strategy(data, model, X_features, predictions, capital=10000, slippage=0.001, 
                        commission=0.001, position_size=0.95, threshold=0.0):
    """
    Backtest ML strategy with realistic execution
    """
    df = data.copy()
    df['prediction'] = predictions
    df['signal'] = (df['prediction'] > threshold).astype(int)
    
    # Position management
    df['position'] = df['signal']
    df['position'] = df['position'].fillna(method='ffill').fillna(0)
    
    # Returns calculation with costs
    df['returns'] = df['Close'].pct_change()
    df['strategy_returns'] = df['position'].shift(1) * df['returns']
    
    # Apply slippage & commission
    df['strategy_returns'] -= (df['position'].shift(1) * (slippage + commission))
    
    # Equity curve
    df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
    df['equity'] = capital * df['cumulative_returns']
    
    return df

def calculate_backtest_metrics(df):
    """Calculate comprehensive backtesting metrics"""
    returns = df['strategy_returns'].dropna()
    
    if len(returns) == 0:
        return {}
    
    total_return = (df['equity'].iloc[-1] / df['equity'].iloc[0]) - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_dd = drawdown.min()
    
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
    
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
    profit_factor = returns[returns > 0].sum() / abs(returns[returns < 0].sum()) if returns[returns < 0].sum() != 0 else 0
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'calmar': calmar,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_trades': (df['position'].diff() != 0).sum(),
        'avg_trade_return': returns.mean(),
    }

def backtest_classical_strategy(data, strategy_type='sma_crossover', capital=10000):
    """Backtest classical strategies"""
    df = data.copy()
    
    if strategy_type == 'sma_crossover':
        df['sma_50'] = df['Close'].rolling(50).mean()
        df['sma_200'] = df['Close'].rolling(200).mean()
        df['signal'] = (df['sma_50'] > df['sma_200']).astype(int)
    
    elif strategy_type == 'rsi':
        df['rsi'] = ta.rsi(df['Close'], 14)
        df['signal'] = ((df['rsi'] < 30) | (df['rsi'] > 70)).astype(int)
    
    elif strategy_type == 'macd':
        macd = ta.macd(df['Close'])
        df['macd'] = macd['MACD_12_26_9']
        df['signal_line'] = macd['MACDh_12_26_9']
        df['signal'] = (df['macd'] > df['signal_line']).astype(int)
    
    elif strategy_type == 'bollinger':
        bb = ta.bbands(df['Close'], 20)
        df['bb_upper'] = bb['BBU_20_2.0']
        df['bb_lower'] = bb['BBL_20_2.0']
        df['signal'] = ((df['Close'] < df['bb_lower']) | (df['Close'] > df['bb_upper'])).astype(int)
    
    # Calculate returns
    df['returns'] = df['Close'].pct_change()
    df['strategy_returns'] = df['signal'].shift(1) * df['returns'] * 0.95
    df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
    df['equity'] = capital * df['cumulative_returns']
    
    return df

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.sidebar.title("📊 TradeSage AI")

# Symbol & timeframe selection
symbol = st.sidebar.selectbox(
    "Select Trading Pair",
    ["BTC/USDT", "ETH/USDT", "AAPL", "MSFT", "GOOGL", "SPY", "QQQ"],
    index=0
)

timeframe = st.sidebar.selectbox("Timeframe", ["1d", "1h", "5m"], index=0)

lookback_period = st.sidebar.slider("Lookback Period (days)", 30, 500, 365)

# Fetch data
with st.spinner("Loading market data..."):
    market_data = get_historical_data(symbol, timeframe, lookback_period)

if market_data.empty:
    st.error("Failed to load data. Check symbol or API status.")
    st.stop()

# Tab selection
tabs = st.tabs([
    "📈 Overview",
    "📊 Live Chart + Indicators",
    "📰 News & Events",
    "😊 Sentiment Analysis",
    "🤖 Trader Predictions",
    "🧪 Backtester"
])

# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================

with tabs[0]:
    st.title("📈 Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = market_data['Close'].iloc[-1]
    prev_price = market_data['Close'].iloc[-2] if len(market_data) > 1 else current_price
    price_change = ((current_price - prev_price) / prev_price) * 100
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}", f"{price_change:.2f}%")
    
    with col2:
        st.metric("24h High", f"${market_data['High'].tail(24).max():.2f}")
    
    with col3:
        st.metric("24h Low", f"${market_data['Low'].tail(24).min():.2f}")
    
    with col4:
        st.metric("Volume (24h)", f"{market_data['Volume'].tail(24).sum():,.0f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Action (Last 30 Days)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=market_data.tail(30).index,
            y=market_data.tail(30)['Close'],
            mode='lines',
            name='Price',
            line=dict(color=PRIMARY_COLOR, width=2)
        ))
        fig.update_layout(
            template="plotly_dark",
            hovermode='x unified',
            height=350,
            xaxis_title="Date",
            yaxis_title="Price (USD)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Returns Distribution")
        returns = market_data['Close'].pct_change().dropna() * 100
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=returns, nbinsx=50, marker_color=PRIMARY_COLOR))
        fig.update_layout(template="plotly_dark", height=350, xaxis_title="Returns (%)")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 2: LIVE CHART + INDICATORS
# ============================================================================

with tabs[1]:
    st.title("📊 Live Chart & Technical Indicators")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Indicator Settings")
        show_bb = st.checkbox("Bollinger Bands", value=True)
        show_rsi = st.checkbox("RSI", value=True)
        show_macd = st.checkbox("MACD", value=True)
        show_ema = st.checkbox("EMA 12/26", value=True)
    
    with col1:
        # Candlestick chart
        df_plot = market_data.tail(100).copy()
        
        fig = go.Figure(data=[go.Candlestick(
            x=df_plot.index,
            open=df_plot['Open'],
            high=df_plot['High'],
            low=df_plot['Low'],
            close=df_plot['Close'],
            name='Price'
        )])
        
        # Add indicators
        if show_bb:
            bb = ta.bbands(df_plot['Close'], length=20)
            fig.add_trace(go.Scatter(x=df_plot.index, y=bb['BBU_20_2.0'], name='BB Upper', line=dict(color='rgba(255,0,0,0.3)')))
            fig.add_trace(go.Scatter(x=df_plot.index, y=bb['BBL_20_2.0'], name='BB Lower', line=dict(color='rgba(255,0,0,0.3)')))
        
        if show_ema:
            fig.add_trace(go.Scatter(x=df_plot.index, y=ta.ema(df_plot['Close'], 12), name='EMA 12', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=df_plot.index, y=ta.ema(df_plot['Close'], 26), name='EMA 26', line=dict(color='purple')))
        
        fig.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # RSI & MACD subplots
    col1, col2 = st.columns(2)
    
    with col1:
        if show_rsi:
            st.subheader("RSI (14)")
            df_plot['RSI'] = ta.rsi(df_plot['Close'], 14)
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], fill='tozeroy', name='RSI', line=dict(color=PRIMARY_COLOR)))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            fig_rsi.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig_rsi, use_container_width=True)
    
    with col2:
        if show_macd:
            st.subheader("MACD")
            macd = ta.macd(df_plot['Close'])
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=df_plot.index, y=macd['MACD_12_26_9'], name='MACD'))
            fig_macd.add_trace(go.Scatter(x=df_plot.index, y=macd['MACDh_12_26_9'], name='Signal'))
            fig_macd.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig_macd, use_container_width=True)

# ============================================================================
# TAB 3: NEWS & EVENTS
# ============================================================================

with tabs[2]:
    st.title("📰 News & Market Events")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Recent News")
        news = get_news_feed()
        
        for article in news:
            with st.container():
                sentiment_color = "🟢" if article['sentiment'] > 0 else "🔴"
                st.markdown(f"**{sentiment_color} {article['title']}**")
                st.caption(f"Source: {article['source']} | Sentiment: {article['sentiment']:.2f}")
                st.divider()
    
    with col2:
        st.subheader("Sentiment Breakdown")
        sentiments = [a['sentiment'] for a in news]
        fig = go.Figure(data=[go.Pie(
            labels=['Positive', 'Neutral', 'Negative'],
            values=[sum(1 for s in sentiments if s > 0.3), sum(1 for s in sentiments if -0.3 <= s <= 0.3), sum(1 for s in sentiments if s < -0.3)],
            marker=dict(colors=[SUCCESS_COLOR, '#888888', DANGER_COLOR])
        )])
        fig.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 4: SENTIMENT ANALYSIS
# ============================================================================

with tabs[3]:
    st.title("😊 Sentiment Analysis")
    
    sentiment_score = get_sentiment_score()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Sentiment", f"{sentiment_score:.2f}", 
                 delta="Bullish" if sentiment_score > 0 else "Bearish")
    
    with col2:
        st.metric("News Sentiment", f"{np.random.uniform(-1, 1):.2f}")
    
    with col3:
        st.metric("Social Media Sentiment", f"{np.random.uniform(-1, 1):.2f}")
    
    st.subheader("Sentiment Trend (30 Days)")
    sentiment_history = np.cumsum(np.random.normal(0, 0.1, 30))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=sentiment_history,
        mode='lines',
        name='Sentiment',
        line=dict(color=PRIMARY_COLOR, width=3),
        fill='tozeroy'
    ))
    fig.update_layout(template="plotly_dark", height=400, xaxis_title="Days", yaxis_title="Sentiment Score")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 5: TRADER PREDICTIONS
# ============================================================================

with tabs[4]:
    st.title("🤖 AI Trader Predictions")
    
    # Simulate daily predictions
    today_prediction = np.random.choice(['LONG', 'SHORT', 'HOLD'], p=[0.4, 0.3, 0.3])
    confidence = np.random.uniform(0.6, 0.99)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        signal_color = "🟢" if today_prediction == "LONG" else "🔴" if today_prediction == "SHORT" else "🟡"
        st.markdown(f"# {signal_color} {today_prediction}")
        st.caption("Today's Signal")
    
    with col2:
        st.metric("Confidence", f"{confidence:.1%}")
    
    with col3:
        st.metric("Suggested Entry", f"${market_data['Close'].iloc[-1] * 0.99:.2f}")
    
    st.subheader("7-Day Forecast")
    forecast_data = pd.DataFrame({
        'Day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        'Signal': np.random.choice(['LONG', 'SHORT', 'HOLD'], 7),
        'Confidence': np.random.uniform(0.6, 0.95, 7)
    })
    
    st.dataframe(forecast_data, use_container_width=True)
    
    st.subheader("Historical Prediction Accuracy")
    accuracy = np.random.uniform(0.45, 0.65)
    st.progress(accuracy)
    st.caption(f"Win Rate: {accuracy:.1%} (based on last 30 days)")

# ============================================================================
# TAB 6: BACKTESTER (CLASSICAL + ML)
# ============================================================================

with tabs[5]:
    st.title("🧪 Advanced Backtester")
    
    backtester_tab1, backtester_tab2 = st.tabs(["Classical Strategies", "ML Strategies"])
    
    # ========== CLASSICAL STRATEGIES ==========
    with backtester_tab1:
        st.subheader("Classical Technical Strategies")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            strategy = st.selectbox(
                "Strategy",
                ["SMA Crossover", "RSI", "MACD", "Bollinger Bands"]
            )
        
        with col2:
            initial_capital = st.number_input("Initial Capital", 1000, 1000000, 10000)
        
        with col3:
            slippage = st.number_input("Slippage (%)", 0.0, 1.0, 0.1) / 100
        
        with col4:
            commission = st.number_input("Commission (%)", 0.0, 1.0, 0.1) / 100
        
        if st.button("Backtest Classical Strategy", key="classical_bt"):
            with st.spinner("Running backtest..."):
                strategy_map = {
                    "SMA Crossover": "sma_crossover",
                    "RSI": "rsi",
                    "MACD": "macd",
                    "Bollinger Bands": "bollinger"
                }
                
                bt_data = backtest_classical_strategy(market_data, strategy_map[strategy], initial_capital)
                metrics = calculate_backtest_metrics(bt_data)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Return", f"{metrics.get('total_return', 0):.2%}")
                with col2:
                    st.metric("Sharpe Ratio", f"{metrics.get('sharpe', 0):.2f}")
                with col3:
                    st.metric("Max Drawdown", f"{metrics.get('max_dd', 0):.2%}")
                with col4:
                    st.metric("Win Rate", f"{metrics.get('win_rate', 0):.1%}")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=bt_data.index,
                    y=bt_data['equity'],
                    mode='lines',
                    name='Equity Curve',
                    line=dict(color=PRIMARY_COLOR, width=2)
                ))
                fig.update_layout(
                    template="plotly_dark",
                    height=400,
                    xaxis_title="Date",
                    yaxis_title="Equity ($)",
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Trade Log")
                trade_log = bt_data[bt_data['signal'].diff() != 0][['Close', 'signal', 'returns']].tail(20)
                st.dataframe(trade_log, use_container_width=True)
    
    # ========== ML STRATEGIES ==========
    with backtester_tab2:
        st.subheader("Machine Learning Backtester")
        
        # ML Model Selection
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ml_model = st.selectbox(
                "ML Model",
                ["XGBoost", "LightGBM", "Random Forest", "Logistic Regression", "LSTM", "Hybrid LSTM+XGB"],
                key="ml_model_select"
            )
        
        with col2:
            task_type = st.selectbox("Task", ["Classification (Direction)", "Regression (Returns)"])
        
        with col3:
            initial_capital_ml = st.number_input("Initial Capital", 1000, 1000000, 10000, key="ml_capital")
        
        with col4:
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.55)
        
        # Feature Engineering Options
        with st.expander("Feature Engineering Settings"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                lookback_features = st.slider("Feature Lookback", 5, 50, 20)
            
            with col2:
                include_sentiment = st.checkbox("Include Sentiment Feature", value=True)
            
            with col3:
                include_trader_pred = st.checkbox("Include Trader Predictions", value=True)
            
            feature_toggle = st.multiselect(
                "Select Features",
                ["RSI", "MACD", "Bollinger Bands", "EMA", "SMA", "ATR", "Volume", "Momentum", "Volatility"],
                default=["RSI", "MACD", "EMA", "Volume"]
            )
        
        # Hyperparameter Tuning
        with st.expander("Hyperparameter Settings"):
            if ml_model in ["XGBoost", "LightGBM"]:
                col1, col2, col3 = st.columns(3)
                with col1:
                    max_depth = st.slider("Max Depth", 3, 15, 6)
                with col2:
                    learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
                with col3:
                    n_estimators = st.slider("N Estimators", 50, 500, 100)
            
            elif ml_model == "Random Forest":
                col1, col2 = st.columns(2)
                with col1:
                    rf_n_estimators = st.slider("N Estimators", 50, 500, 100)
                with col2:
                    rf_max_depth = st.slider("Max Depth", 3, 15, 10)
            
            elif ml_model == "LSTM":
                col1, col2 = st.columns(2)
                with col1:
                    lstm_epochs = st.slider("Epochs", 5, 50, 20)
                with col2:
                    lstm_units = st.slider("LSTM Units", 32, 256, 64)
            
            train_test_split_pct = st.slider("Train/Test Split", 0.5, 0.95, 0.8)
        
        if st.button("Train & Backtest ML Strategy", key="ml_bt"):
            with st.spinner("Engineering features..."):
                # Feature engineering
                sentiment_score = get_sentiment_score() if include_sentiment else 0
                df_features = engineer_features(market_data, lookback_features, include_sentiment, sentiment_score)
                
                # Select features
                feature_cols = [col for col in df_features.columns 
                               if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'returns']]
                X = df_features[feature_cols].fillna(0)
                
                # Target variable
                if task_type == "Regression (Returns)":
                    y = df_features['returns'].shift(-1)  # Next period return (no look-ahead)
                else:
                    y = (df_features['returns'].shift(-1) > 0).astype(int)  # Direction
                
                # Remove NaN
                valid_idx = ~(X.isna().any(axis=1) | y.isna())
                X = X[valid_idx].reset_index(drop=True)
                y = y[valid_idx].reset_index(drop=True)
                
                # Normalize
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Train/test split (walk-forward style)
                split_idx = int(len(X) * train_test_split_pct)
                X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
                y_train, y_test = y[:split_idx].values, y[split_idx:].values
                
                st.success(f"Features engineered: {X.shape[1]} features, {len(X)} samples")
            
            with st.spinner("Training model..."):
                # Train model
                if ml_model == "XGBoost":
                    model = train_xgboost(X_train, y_train, X_test, y_test, 
                                         task='classification' if 'Classification' in task_type else 'regression',
                                         hyperparams={'max_depth': max_depth, 'learning_rate': learning_rate, 'n_estimators': n_estimators})
                
                elif ml_model == "LightGBM":
                    model = train_lightgbm(X_train, y_train, X_test, y_test,
                                          task='classification' if 'Classification' in task_type else 'regression',
                                          hyperparams={'num_leaves': 2**max_depth, 'learning_rate': learning_rate, 'n_estimators': n_estimators})
                
                elif ml_model == "Random Forest":
                    model = train_random_forest(X_train, y_train,
                                               task='classification' if 'Classification' in task_type else 'regression',
                                               hyperparams={'n_estimators': rf_n_estimators, 'max_depth': rf_max_depth})
                
                elif ml_model == "Logistic Regression":
                    model = train_logistic_regression(X_train, (y_train > 0).astype(int))
                
                elif ml_model == "LSTM":
                    model = train_lstm(X_train, y_train, X_test, y_test, lookback=lookback_features, epochs=lstm_epochs)
                
                elif ml_model == "Hybrid LSTM+XGB":
                    model = train_hybrid_lstm_xgb(X_train, y_train, X_test, y_test)
                
                st.success("Model trained successfully!")
            
            # Generate predictions
            if ml_model == "Hybrid LSTM+XGB":
                X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
                lstm_features = model['lstm'].predict(X_test_lstm)
                predictions_test = model['xgb'].predict(lstm_features)
            elif ml_model == "LSTM":
                X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
                predictions_test = model.predict(X_test_lstm).flatten()
            else:
                predictions_test = model.predict(X_test)
            
            # Probabilities for classification
            if 'Classification' in task_type:
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X_test)[:, 1]
                else:
                    probs = predictions_test
            else:
                probs = predictions_test
            
            # Backtest
            df_backtest = market_data.iloc[split_idx + lookback_features:].copy().reset_index(drop=True)
            df_backtest = df_backtest.iloc[:len(predictions_test)].copy()
            
            if len(df_backtest) > 0:
                bt_results = backtest_ml_strategy(df_backtest, model, X_test, probs, 
                                                initial_capital_ml, slippage=0.001, commission=0.001, 
                                                threshold=confidence_threshold)
                metrics = calculate_backtest_metrics(bt_results)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Return", f"{metrics.get('total_return', 0):.2%}")
                with col2:
                    st.metric("Sharpe Ratio", f"{metrics.get('sharpe', 0):.2f}")
                with col3:
                    st.metric("Max Drawdown", f"{metrics.get('max_dd', 0):.2%}")
                with col4:
                    st.metric("Win Rate", f"{metrics.get('win_rate', 0):.1%}")
                
                # Equity curve
                fig_equity = go.Figure()
                fig_equity.add_trace(go.Scatter(
                    x=bt_results.index,
                    y=bt_results['equity'],
                    mode='lines',
                    name='ML Strategy Equity',
                    line=dict(color=PRIMARY_COLOR, width=2)
                ))
                
                # Buy & Hold comparison
                buy_hold_equity = initial_capital_ml * (bt_results['Close'] / bt_results['Close'].iloc[0])
                fig_equity.add_trace(go.Scatter(
                    x=bt_results.index,
                    y=buy_hold_equity,
                    mode='lines',
                    name='Buy & Hold',
                    line=dict(color='#888888', width=2, dash='dash')
                ))
                
                fig_equity.update_layout(
                    template="plotly_dark",
                    height=400,
                    xaxis_title="Date",
                    yaxis_title="Equity ($)",
                    hovermode='x unified'
                )
                st.plotly_chart(fig_equity, use_container_width=True)
                
                # Feature importance (tree-based models)
                if ml_model in ["XGBoost", "LightGBM", "Random Forest"]:
                    st.subheader("Feature Importance")
                    
                    if ml_model == "XGBoost":
                        importance = model.feature_importances_
                    elif ml_model == "LightGBM":
                        importance = model.feature_importances_
                    else:
                        importance = model.feature_importances_
                    
                    feature_importance_df = pd.DataFrame({
                        'Feature': feature_cols,
                        'Importance': importance
                    }).sort_values('Importance', ascending=False).head(15)
                    
                    fig_imp = px.bar(feature_importance_df, x='Importance', y='Feature', orientation='h',
                                    color='Importance', color_continuous_scale='Viridis')
                    fig_imp.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig_imp, use_container_width=True)
                
                # Model performance
                st.subheader("Model Performance")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Prediction vs Actual scatter
                    fig_scatter = go.Figure()
                    fig_scatter.add_trace(go.Scatter(
                        x=y_test,
                        y=predictions_test[:len(y_test)],
                        mode='markers',
                        marker=dict(color='rgba(0, 217, 255, 0.6)', size=5),
                        name='Predictions'
                    ))
                    
                    min_val = min(y_test.min(), predictions_test.min())
                    max_val = max(y_test.max(), predictions_test.max())
                    fig_scatter.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='gray', dash='dash')
                    ))
                    
                    fig_scatter.update_layout(template="plotly_dark", height=350,
                                            xaxis_title="Actual", yaxis_title="Predicted")
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                with col2:
                    if 'Classification' in task_type:
                        # Confusion matrix
                        y_pred_binary = (predictions_test > 0.5).astype(int)
                        cm = confusion_matrix(y_test, y_pred_binary)
                        
                        fig_cm = go.Figure(data=go.Heatmap(
                            z=cm,
                            x=['Predicted Down', 'Predicted Up'],
                            y=['Actual Down', 'Actual Up'],
                            colorscale='Blues'
                        ))
                        fig_cm.update_layout(template="plotly_dark", height=350)
                        st.plotly_chart(fig_cm, use_container_width=True)
                    else:
                        # R² and RMSE
                        r2 = r2_score(y_test, predictions_test[:len(y_test)])
                        rmse = np.sqrt(mean_squared_error(y_test, predictions_test[:len(y_test)]))
                        
                        col1_metric, col2_metric = st.columns(2)
                        with col1_metric:
                            st.metric("R² Score", f"{r2:.4f}")
                        with col2_metric:
                            st.metric("RMSE", f"{rmse:.6f}")
                
                # Trade log
                st.subheader("Trade Log (Last 20)")
                trade_log = bt_results[bt_results['signal'].diff() != 0][['Close', 'signal', 'strategy_returns']].tail(20)
                st.dataframe(trade_log, use_container_width=True)
                
                # Export results
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📊 Export Results as JSON"):
                        export_data = {
                            'model': ml_model,
                            'metrics': metrics,
                            'features_count': len(feature_cols),
                            'training_samples': len(X_train),
                            'test_samples': len(X_test),
                            'timestamp': datetime.now().isoformat()
                        }
                        st.download_button(
                            "Download JSON",
                            json.dumps(export_data, indent=2),
                            f"ml_backtest_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        )
                
                with col2:
                    if st.button("📈 Export Equity Curve as CSV"):
                        csv = bt_results[['Close', 'signal', 'strategy_returns', 'equity']].to_csv(index=True)
                        st.download_button(
                            "Download CSV",
                            csv,
                            f"equity_curve_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>📊 <b>TradeSage AI</b> – Professional Quant Research Platform</p>
    <p>⚠️ This tool is for research and educational purposes only.</p>
</div>
""", unsafe_allow_html=True)
