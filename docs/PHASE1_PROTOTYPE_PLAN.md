# Phase 1: Prototype Implementation Plan

**Goal:** Build a working prediction engine focused on accuracy, with minimal UI.
**User:** Single user (you)
**Timeline:** 4-6 weeks
**Focus:** Get predictions right first, UI later.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     PROTOTYPE ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    LOCAL MACHINE (Your PC)                       │    │
│  │                                                                  │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │    │
│  │  │   Python     │  │   SQLite     │  │   React      │           │    │
│  │  │   ML Engine  │  │   Database   │  │   Dashboard  │           │    │
│  │  │              │  │              │  │   (Minimal)  │           │    │
│  │  │  - Training  │  │  - Prices    │  │              │           │    │
│  │  │  - Inference │  │  - Preds     │  │  - View preds│           │    │
│  │  │  - Patterns  │  │  - Patterns  │  │  - Accuracy  │           │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │    │
│  │         │                  │                  │                  │    │
│  │         └──────────────────┼──────────────────┘                  │    │
│  │                            │                                     │    │
│  │  ┌─────────────────────────▼─────────────────────────────┐      │    │
│  │  │              FastAPI Backend (Local)                   │      │    │
│  │  │  - REST endpoints for UI                               │      │    │
│  │  │  - Trigger training/predictions                        │      │    │
│  │  └───────────────────────────────────────────────────────┘      │    │
│  │                                                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  External: Alpha Vantage / Yahoo Finance API (Free)                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack (Simplified for Prototype)

| Component | Technology | Why |
|-----------|------------|-----|
| **ML/Training** | Python + PyTorch + XGBoost | Best ML ecosystem, GPU support |
| **Database** | SQLite | Zero config, single file, sufficient for 1 user |
| **Backend API** | FastAPI | Fast, async, great for ML serving |
| **Frontend** | React + Vite (minimal) | Simple dashboard only |
| **Data Source** | yfinance (free) | Unlimited, reliable |
| **Charts** | Plotly (Python) | Quick visualization during dev |

**NOT needed for prototype:**
- Azure services
- Cosmos DB / Azure SQL
- .NET backend
- Authentication
- Docker/containers

---

## Project Structure

```
PricePrediction/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetcher.py          # Download price data
│   │   ├── database.py         # SQLite operations
│   │   └── features.py         # Feature engineering
│   │
│   ├── patterns/
│   │   ├── __init__.py
│   │   ├── candlestick.py      # Candlestick pattern detection
│   │   ├── indicators.py       # RSI, MACD, etc.
│   │   ├── support_resistance.py
│   │   └── detector.py         # Unified pattern detector
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── xgboost_model.py    # XGBoost predictor
│   │   ├── lstm_model.py       # LSTM predictor
│   │   ├── ensemble.py         # Combine models
│   │   └── trainer.py          # Training orchestrator
│   │
│   ├── prediction/
│   │   ├── __init__.py
│   │   ├── predictor.py        # Make predictions
│   │   └── evaluator.py        # Track accuracy
│   │
│   └── api/
│       ├── __init__.py
│       └── main.py             # FastAPI app
│
├── ui/                         # Minimal React app
│   ├── src/
│   │   ├── App.tsx
│   │   └── components/
│   │       ├── PredictionTable.tsx
│   │       ├── AccuracyChart.tsx
│   │       └── StockDetail.tsx
│   └── package.json
│
├── notebooks/                  # Jupyter for experiments
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_backtesting.ipynb
│
├── data/
│   └── priceprediction.db      # SQLite database
│
├── models/                     # Saved model files
│   └── .gitkeep
│
├── scripts/
│   ├── fetch_data.py           # Daily data fetch
│   ├── train_models.py         # Training script
│   └── run_predictions.py      # Generate predictions
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Implementation Plan

### Week 1: Data Foundation

#### Day 1-2: Project Setup
```bash
# Create project structure
mkdir -p src/{data,patterns,models,prediction,api}
mkdir -p ui/src/components
mkdir -p notebooks data models scripts

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install xgboost pandas numpy scikit-learn yfinance
pip install fastapi uvicorn sqlalchemy
pip install ta pandas-ta  # Technical analysis
pip install jupyter plotly matplotlib seaborn
pip install mlflow  # Experiment tracking
```

#### Day 3-4: Data Fetching & Storage

**File: `src/data/fetcher.py`**
```python
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class DataFetcher:
    def __init__(self):
        self.default_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
            'JPM', 'V', 'WMT', 'SPY', 'QQQ'  # Start with 12 liquid stocks
        ]

    def fetch_history(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """Fetch historical OHLCV data"""
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        df['symbol'] = symbol
        return df

    def fetch_all(self, symbols: list = None) -> pd.DataFrame:
        """Fetch data for all symbols"""
        symbols = symbols or self.default_symbols
        all_data = []

        for symbol in symbols:
            try:
                df = self.fetch_history(symbol)
                all_data.append(df)
                print(f"Fetched {symbol}: {len(df)} rows")
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")

        return pd.concat(all_data, ignore_index=True)
```

**File: `src/data/database.py`**
```python
from sqlalchemy import create_engine, Column, String, Float, Date, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd

Base = declarative_base()

class Price(Base):
    __tablename__ = 'prices'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, index=True)
    date = Column(Date, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, index=True)
    prediction_date = Column(Date)  # When prediction was made
    target_date = Column(Date)      # What date we're predicting
    predicted_direction = Column(String)  # up/down/neutral
    predicted_change = Column(Float)      # % change
    confidence = Column(Float)
    model_name = Column(String)
    actual_direction = Column(String, nullable=True)  # Filled later
    actual_change = Column(Float, nullable=True)
    is_correct = Column(Integer, nullable=True)  # 1/0

class Pattern(Base):
    __tablename__ = 'patterns'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, index=True)
    date = Column(Date)
    pattern_type = Column(String)
    pattern_name = Column(String)
    signal = Column(String)  # bullish/bearish
    strength = Column(Float)

class Database:
    def __init__(self, db_path: str = "data/priceprediction.db"):
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def save_prices(self, df: pd.DataFrame):
        df.to_sql('prices', self.engine, if_exists='append', index=False)

    def get_prices(self, symbol: str, days: int = 500) -> pd.DataFrame:
        query = f"""
            SELECT * FROM prices
            WHERE symbol = '{symbol}'
            ORDER BY date DESC
            LIMIT {days}
        """
        return pd.read_sql(query, self.engine)

    def save_prediction(self, prediction: dict):
        session = self.Session()
        pred = Prediction(**prediction)
        session.add(pred)
        session.commit()
        session.close()

    def get_prediction_accuracy(self, days: int = 30) -> dict:
        query = f"""
            SELECT
                model_name,
                COUNT(*) as total,
                SUM(is_correct) as correct,
                AVG(is_correct) * 100 as accuracy
            FROM predictions
            WHERE is_correct IS NOT NULL
            AND prediction_date >= date('now', '-{days} days')
            GROUP BY model_name
        """
        return pd.read_sql(query, self.engine).to_dict('records')
```

#### Day 5-7: Feature Engineering

**File: `src/data/features.py`**
```python
import pandas as pd
import numpy as np
import pandas_ta as ta

class FeatureEngineer:
    def __init__(self):
        self.feature_names = []

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features for ML model"""
        df = df.copy()

        # Ensure sorted by date
        df = df.sort_values('date').reset_index(drop=True)

        # === PRICE FEATURES ===
        # Returns at various horizons
        for period in [1, 2, 3, 5, 10, 20]:
            df[f'return_{period}d'] = df['close'].pct_change(period)

        # Price relative to moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1

        # EMA
        for period in [12, 26, 50]:
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()

        # === VOLATILITY FEATURES ===
        for period in [5, 10, 20]:
            df[f'volatility_{period}d'] = df['return_1d'].rolling(period).std()

        # ATR
        df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)

        # Bollinger Bands
        bb = ta.bbands(df['close'], length=20)
        df['bb_upper'] = bb['BBU_20_2.0']
        df['bb_lower'] = bb['BBL_20_2.0']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # === MOMENTUM FEATURES ===
        # RSI
        df['rsi_14'] = ta.rsi(df['close'], length=14)
        df['rsi_7'] = ta.rsi(df['close'], length=7)

        # MACD
        macd = ta.macd(df['close'])
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_hist'] = macd['MACDh_12_26_9']

        # Stochastic
        stoch = ta.stoch(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch['STOCHk_14_3_3']
        df['stoch_d'] = stoch['STOCHd_14_3_3']

        # ROC (Rate of Change)
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = ta.roc(df['close'], length=period)

        # Williams %R
        df['willr'] = ta.willr(df['high'], df['low'], df['close'])

        # CCI
        df['cci'] = ta.cci(df['high'], df['low'], df['close'])

        # ADX (trend strength)
        adx = ta.adx(df['high'], df['low'], df['close'])
        df['adx'] = adx['ADX_14']
        df['di_plus'] = adx['DMP_14']
        df['di_minus'] = adx['DMN_14']

        # === VOLUME FEATURES ===
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']

        # OBV
        df['obv'] = ta.obv(df['close'], df['volume'])
        df['obv_sma'] = df['obv'].rolling(20).mean()
        df['obv_ratio'] = df['obv'] / df['obv_sma']

        # === PATTERN FEATURES ===
        # Higher highs / Lower lows
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        df['higher_close'] = (df['close'] > df['close'].shift(1)).astype(int)

        # Consecutive up/down days
        df['up_streak'] = df['higher_close'].groupby(
            (df['higher_close'] != df['higher_close'].shift()).cumsum()
        ).cumsum() * df['higher_close']

        df['down_streak'] = (1 - df['higher_close']).groupby(
            ((1-df['higher_close']) != (1-df['higher_close']).shift()).cumsum()
        ).cumsum() * (1 - df['higher_close'])

        # Gap
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

        # Candle body
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
        df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']

        # === TIME FEATURES ===
        if 'date' in df.columns:
            df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
            df['month'] = pd.to_datetime(df['date']).dt.month
            df['day_of_month'] = pd.to_datetime(df['date']).dt.day

        # Store feature names (excluding non-features)
        exclude = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                   'dividends', 'stock_splits']
        self.feature_names = [c for c in df.columns if c not in exclude]

        return df

    def create_target(self, df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
        """Create prediction target"""
        df = df.copy()

        # Future return (what we want to predict)
        df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1

        # Classification target
        threshold = 0.02  # 2% move threshold
        df['target'] = 1  # neutral
        df.loc[df['future_return'] > threshold, 'target'] = 2   # up
        df.loc[df['future_return'] < -threshold, 'target'] = 0  # down

        return df

    def get_feature_matrix(self, df: pd.DataFrame) -> tuple:
        """Get X, y for training"""
        feature_cols = [c for c in self.feature_names if c in df.columns]
        X = df[feature_cols].values
        y = df['target'].values if 'target' in df.columns else None
        return X, y, feature_cols
```

---

### Week 2: Pattern Detection

#### Day 8-10: Candlestick & Indicator Patterns

**File: `src/patterns/candlestick.py`**
```python
import pandas as pd
import numpy as np

class CandlestickDetector:
    def __init__(self, body_threshold=0.001):
        self.body_threshold = body_threshold

    def detect_all(self, df: pd.DataFrame) -> list:
        """Detect all candlestick patterns"""
        patterns = []

        for i in range(2, len(df)):
            detected = self._detect_at_index(df, i)
            patterns.extend(detected)

        return patterns

    def _detect_at_index(self, df: pd.DataFrame, i: int) -> list:
        """Detect patterns at a specific index"""
        patterns = []
        c = df.iloc[i]      # Current candle
        p1 = df.iloc[i-1]   # Previous candle
        p2 = df.iloc[i-2]   # 2 candles ago

        # Helper functions
        def body(candle):
            return abs(candle['close'] - candle['open'])

        def range_(candle):
            return candle['high'] - candle['low']

        def is_bullish(candle):
            return candle['close'] > candle['open']

        def upper_wick(candle):
            return candle['high'] - max(candle['open'], candle['close'])

        def lower_wick(candle):
            return min(candle['open'], candle['close']) - candle['low']

        date = df.index[i] if isinstance(df.index[i], pd.Timestamp) else df.iloc[i].get('date')

        # === SINGLE CANDLE PATTERNS ===

        # Doji
        if body(c) < 0.1 * range_(c) and range_(c) > 0:
            patterns.append({
                'date': date, 'pattern': 'doji', 'signal': 'neutral',
                'strength': 1 - (body(c) / range_(c))
            })

        # Hammer (bullish)
        if (lower_wick(c) > 2 * body(c) and
            upper_wick(c) < 0.3 * body(c) and
            body(c) > 0):
            patterns.append({
                'date': date, 'pattern': 'hammer', 'signal': 'bullish',
                'strength': min(lower_wick(c) / body(c) / 3, 1.0)
            })

        # Shooting Star (bearish)
        if (upper_wick(c) > 2 * body(c) and
            lower_wick(c) < 0.3 * body(c) and
            body(c) > 0):
            patterns.append({
                'date': date, 'pattern': 'shooting_star', 'signal': 'bearish',
                'strength': min(upper_wick(c) / body(c) / 3, 1.0)
            })

        # Marubozu (strong momentum)
        if body(c) > 0.95 * range_(c) and range_(c) > 0:
            signal = 'bullish' if is_bullish(c) else 'bearish'
            patterns.append({
                'date': date, 'pattern': 'marubozu', 'signal': signal,
                'strength': body(c) / range_(c)
            })

        # === TWO CANDLE PATTERNS ===

        # Bullish Engulfing
        if (not is_bullish(p1) and is_bullish(c) and
            c['open'] < p1['close'] and c['close'] > p1['open']):
            patterns.append({
                'date': date, 'pattern': 'bullish_engulfing', 'signal': 'bullish',
                'strength': body(c) / body(p1) if body(p1) > 0 else 0.8
            })

        # Bearish Engulfing
        if (is_bullish(p1) and not is_bullish(c) and
            c['open'] > p1['close'] and c['close'] < p1['open']):
            patterns.append({
                'date': date, 'pattern': 'bearish_engulfing', 'signal': 'bearish',
                'strength': body(c) / body(p1) if body(p1) > 0 else 0.8
            })

        # === THREE CANDLE PATTERNS ===

        # Morning Star (bullish reversal)
        if (not is_bullish(p2) and body(p2) > body(p1) and
            is_bullish(c) and c['close'] > (p2['open'] + p2['close'])/2):
            patterns.append({
                'date': date, 'pattern': 'morning_star', 'signal': 'bullish',
                'strength': 0.85
            })

        # Evening Star (bearish reversal)
        if (is_bullish(p2) and body(p2) > body(p1) and
            not is_bullish(c) and c['close'] < (p2['open'] + p2['close'])/2):
            patterns.append({
                'date': date, 'pattern': 'evening_star', 'signal': 'bearish',
                'strength': 0.85
            })

        return patterns
```

**File: `src/patterns/indicators.py`**
```python
import pandas as pd
import numpy as np

class IndicatorPatterns:
    def detect_all(self, df: pd.DataFrame) -> list:
        """Detect indicator-based patterns"""
        patterns = []
        patterns.extend(self._detect_ma_crossovers(df))
        patterns.extend(self._detect_rsi_signals(df))
        patterns.extend(self._detect_macd_signals(df))
        patterns.extend(self._detect_divergences(df))
        return patterns

    def _detect_ma_crossovers(self, df: pd.DataFrame) -> list:
        patterns = []

        # Golden Cross (50 crosses above 200)
        if 'sma_50' in df.columns and 'sma_200' in df.columns:
            for i in range(1, len(df)):
                prev_diff = df.iloc[i-1]['sma_50'] - df.iloc[i-1]['sma_200']
                curr_diff = df.iloc[i]['sma_50'] - df.iloc[i]['sma_200']

                if prev_diff <= 0 and curr_diff > 0:
                    patterns.append({
                        'date': df.index[i],
                        'pattern': 'golden_cross',
                        'signal': 'bullish',
                        'strength': 0.9
                    })
                elif prev_diff >= 0 and curr_diff < 0:
                    patterns.append({
                        'date': df.index[i],
                        'pattern': 'death_cross',
                        'signal': 'bearish',
                        'strength': 0.9
                    })

        return patterns

    def _detect_rsi_signals(self, df: pd.DataFrame) -> list:
        patterns = []

        if 'rsi_14' not in df.columns:
            return patterns

        for i in range(1, len(df)):
            rsi = df.iloc[i]['rsi_14']
            prev_rsi = df.iloc[i-1]['rsi_14']

            # Oversold bounce
            if prev_rsi < 30 and rsi >= 30:
                patterns.append({
                    'date': df.index[i],
                    'pattern': 'rsi_oversold_bounce',
                    'signal': 'bullish',
                    'strength': (30 - prev_rsi) / 30
                })

            # Overbought drop
            if prev_rsi > 70 and rsi <= 70:
                patterns.append({
                    'date': df.index[i],
                    'pattern': 'rsi_overbought_drop',
                    'signal': 'bearish',
                    'strength': (prev_rsi - 70) / 30
                })

        return patterns

    def _detect_macd_signals(self, df: pd.DataFrame) -> list:
        patterns = []

        if 'macd' not in df.columns or 'macd_signal' not in df.columns:
            return patterns

        for i in range(1, len(df)):
            prev_diff = df.iloc[i-1]['macd'] - df.iloc[i-1]['macd_signal']
            curr_diff = df.iloc[i]['macd'] - df.iloc[i]['macd_signal']

            if prev_diff <= 0 and curr_diff > 0:
                patterns.append({
                    'date': df.index[i],
                    'pattern': 'macd_bullish_cross',
                    'signal': 'bullish',
                    'strength': 0.75
                })
            elif prev_diff >= 0 and curr_diff < 0:
                patterns.append({
                    'date': df.index[i],
                    'pattern': 'macd_bearish_cross',
                    'signal': 'bearish',
                    'strength': 0.75
                })

        return patterns

    def _detect_divergences(self, df: pd.DataFrame) -> list:
        """Detect RSI divergences"""
        patterns = []

        if 'rsi_14' not in df.columns:
            return patterns

        # Find local peaks and troughs (simplified)
        window = 5

        for i in range(window * 2, len(df) - window):
            # Look for price making new low but RSI making higher low (bullish)
            price_segment = df['close'].iloc[i-window*2:i+1]
            rsi_segment = df['rsi_14'].iloc[i-window*2:i+1]

            price_min_idx = price_segment.idxmin()
            current_price = df.iloc[i]['close']
            current_rsi = df.iloc[i]['rsi_14']

            # If current is near a local low
            if current_price == price_segment.min():
                # Check if there's a previous low with higher RSI
                prev_lows = price_segment[price_segment < price_segment.median()]
                if len(prev_lows) > 1:
                    prev_rsi_at_low = rsi_segment.loc[prev_lows.index[0]]
                    if current_rsi > prev_rsi_at_low + 5:
                        patterns.append({
                            'date': df.index[i],
                            'pattern': 'bullish_divergence',
                            'signal': 'bullish',
                            'strength': 0.8
                        })

        return patterns
```

**File: `src/patterns/support_resistance.py`**
```python
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

class SupportResistanceDetector:
    def __init__(self, eps_pct=0.02, min_touches=2):
        self.eps_pct = eps_pct  # Clustering distance as % of price
        self.min_touches = min_touches

    def detect_levels(self, df: pd.DataFrame, num_levels: int = 5) -> list:
        """Find support and resistance levels"""
        prices = df['close'].values
        highs = df['high'].values
        lows = df['low'].values

        # Find local extrema
        peaks = self._find_peaks(highs, window=5)
        troughs = self._find_troughs(lows, window=5)

        all_levels = list(peaks) + list(troughs)

        if len(all_levels) < 2:
            return []

        # Cluster nearby levels
        eps = np.mean(prices) * self.eps_pct
        levels_array = np.array(all_levels).reshape(-1, 1)

        clustering = DBSCAN(eps=eps, min_samples=self.min_touches).fit(levels_array)

        # Get cluster centers
        sr_levels = []
        for label in set(clustering.labels_):
            if label == -1:
                continue
            cluster_points = levels_array[clustering.labels_ == label]
            level = np.mean(cluster_points)
            touches = len(cluster_points)
            sr_levels.append({
                'level': level,
                'touches': touches,
                'strength': touches / len(all_levels),
                'type': 'resistance' if level > prices[-1] else 'support'
            })

        # Sort by strength
        sr_levels.sort(key=lambda x: x['strength'], reverse=True)
        return sr_levels[:num_levels]

    def _find_peaks(self, prices: np.array, window: int) -> list:
        peaks = []
        for i in range(window, len(prices) - window):
            if prices[i] == max(prices[i-window:i+window+1]):
                peaks.append(prices[i])
        return peaks

    def _find_troughs(self, prices: np.array, window: int) -> list:
        troughs = []
        for i in range(window, len(prices) - window):
            if prices[i] == min(prices[i-window:i+window+1]):
                troughs.append(prices[i])
        return troughs

    def check_breakout(self, df: pd.DataFrame, levels: list) -> list:
        """Check if recent price broke through S/R levels"""
        breakouts = []
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]

        for level_info in levels:
            level = level_info['level']

            # Breakout above resistance
            if level_info['type'] == 'resistance':
                if prev_price < level and current_price > level:
                    breakouts.append({
                        'type': 'resistance_breakout',
                        'level': level,
                        'signal': 'bullish',
                        'strength': level_info['strength']
                    })

            # Breakdown below support
            if level_info['type'] == 'support':
                if prev_price > level and current_price < level:
                    breakouts.append({
                        'type': 'support_breakdown',
                        'level': level,
                        'signal': 'bearish',
                        'strength': level_info['strength']
                    })

        return breakouts
```

**File: `src/patterns/detector.py`**
```python
from .candlestick import CandlestickDetector
from .indicators import IndicatorPatterns
from .support_resistance import SupportResistanceDetector
import pandas as pd

class UnifiedPatternDetector:
    def __init__(self):
        self.candlestick = CandlestickDetector()
        self.indicators = IndicatorPatterns()
        self.sr_detector = SupportResistanceDetector()

    def detect_all(self, df: pd.DataFrame) -> dict:
        """Run all pattern detectors"""
        results = {
            'candlestick': self.candlestick.detect_all(df),
            'indicator': self.indicators.detect_all(df),
            'support_resistance': self.sr_detector.detect_levels(df),
        }

        # Check for breakouts
        if results['support_resistance']:
            results['breakouts'] = self.sr_detector.check_breakout(
                df, results['support_resistance']
            )

        # Aggregate signals
        results['summary'] = self._summarize_signals(results)

        return results

    def _summarize_signals(self, results: dict) -> dict:
        """Aggregate all signals into a summary"""
        bullish_score = 0
        bearish_score = 0

        # Count recent patterns (last 5 days)
        for pattern_type in ['candlestick', 'indicator']:
            for pattern in results.get(pattern_type, [])[-10:]:
                if pattern['signal'] == 'bullish':
                    bullish_score += pattern['strength']
                elif pattern['signal'] == 'bearish':
                    bearish_score += pattern['strength']

        # Breakouts have higher weight
        for breakout in results.get('breakouts', []):
            if breakout['signal'] == 'bullish':
                bullish_score += breakout['strength'] * 2
            else:
                bearish_score += breakout['strength'] * 2

        total = bullish_score + bearish_score
        if total == 0:
            return {'bias': 'neutral', 'bullish_pct': 50, 'bearish_pct': 50}

        return {
            'bias': 'bullish' if bullish_score > bearish_score else 'bearish',
            'bullish_pct': round(bullish_score / total * 100, 1),
            'bearish_pct': round(bearish_score / total * 100, 1),
            'strength': abs(bullish_score - bearish_score) / total
        }
```

---

### Week 3: ML Models

#### Day 11-13: XGBoost Model

**File: `src/models/xgboost_model.py`**
```python
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import joblib

class XGBoostPredictor:
    def __init__(self, use_gpu: bool = True):
        self.model = None
        self.feature_names = None
        self.use_gpu = use_gpu

    def train(self, X: np.array, y: np.array, feature_names: list = None):
        """Train XGBoost model with time series cross-validation"""
        self.feature_names = feature_names

        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'mlogloss',
            'tree_method': 'gpu_hist' if self.use_gpu else 'hist',
            'gpu_id': 0,
            'random_state': 42
        }

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=20,
                verbose=False
            )

            val_pred = model.predict(X_val)
            score = accuracy_score(y_val, val_pred)
            scores.append(score)
            print(f"Fold {fold+1}: Accuracy = {score:.4f}")

        print(f"\nMean CV Accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

        # Train final model on all data
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X, y, verbose=False)

        return np.mean(scores)

    def predict(self, X: np.array) -> dict:
        """Make prediction with confidence"""
        probs = self.model.predict_proba(X)

        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            probs = probs[0]

        pred_class = np.argmax(probs, axis=-1)
        confidence = np.max(probs, axis=-1)

        # Map class to direction
        direction_map = {0: 'down', 1: 'neutral', 2: 'up'}

        if isinstance(pred_class, np.ndarray):
            directions = [direction_map[c] for c in pred_class]
        else:
            directions = direction_map[pred_class]

        return {
            'direction': directions,
            'confidence': confidence,
            'probabilities': {
                'down': probs[0] if len(probs.shape) == 1 else probs[:, 0],
                'neutral': probs[1] if len(probs.shape) == 1 else probs[:, 1],
                'up': probs[2] if len(probs.shape) == 1 else probs[:, 2]
            }
        }

    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance"""
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def save(self, path: str):
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names
        }, path)

    def load(self, path: str):
        data = joblib.load(path)
        self.model = data['model']
        self.feature_names = data['feature_names']
```

#### Day 14-16: LSTM Model

**File: `src/models/lstm_model.py`**
```python
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=3, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        # Self-attention on LSTM output
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Use last timestep
        last_hidden = attn_out[:, -1, :]

        output = self.fc(last_hidden)
        return output


class LSTMPredictor:
    def __init__(self, sequence_length: int = 60, use_gpu: bool = True):
        self.sequence_length = sequence_length
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None

        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    def create_sequences(self, X: np.array, y: np.array = None):
        """Create sequences for LSTM"""
        sequences = []
        targets = []

        for i in range(self.sequence_length, len(X)):
            sequences.append(X[i-self.sequence_length:i])
            if y is not None:
                targets.append(y[i])

        X_seq = np.array(sequences)
        y_seq = np.array(targets) if y is not None else None

        return X_seq, y_seq

    def train(self, X: np.array, y: np.array, epochs: int = 50, batch_size: int = 32):
        """Train LSTM model"""
        # Create sequences
        X_seq, y_seq = self.create_sequences(X, y)

        # Normalize features
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()

        # Reshape for scaler
        n_samples, n_timesteps, n_features = X_seq.shape
        X_flat = X_seq.reshape(-1, n_features)
        X_scaled = self.scaler.fit_transform(X_flat).reshape(n_samples, n_timesteps, n_features)

        # Train/val split (time series)
        split = int(0.8 * len(X_scaled))
        X_train, X_val = X_scaled[:split], X_scaled[split:]
        y_train, y_val = y_seq[:split], y_seq[split:]

        # Create DataLoaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Initialize model
        input_size = X_seq.shape[2]
        self.model = LSTMNetwork(input_size=input_size).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        scaler = GradScaler()

        best_val_acc = 0
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()

                with autocast():
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    with autocast():
                        outputs = self.model(X_batch)

                    _, predicted = outputs.max(1)
                    val_total += y_batch.size(0)
                    val_correct += predicted.eq(y_batch).sum().item()

            val_acc = val_correct / val_total
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Val Acc={val_acc:.4f}")

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        self.model.load_state_dict(self.best_state)
        print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")

        return best_val_acc

    def predict(self, X: np.array) -> dict:
        """Make prediction"""
        self.model.eval()

        # Create sequence if needed
        if len(X.shape) == 2:
            X, _ = self.create_sequences(X)

        # Scale
        n_samples, n_timesteps, n_features = X.shape
        X_flat = X.reshape(-1, n_features)
        X_scaled = self.scaler.transform(X_flat).reshape(n_samples, n_timesteps, n_features)

        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            with autocast():
                outputs = self.model(X_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()

        pred_class = np.argmax(probs, axis=1)
        confidence = np.max(probs, axis=1)

        direction_map = {0: 'down', 1: 'neutral', 2: 'up'}

        return {
            'direction': [direction_map[c] for c in pred_class],
            'confidence': confidence,
            'probabilities': {
                'down': probs[:, 0],
                'neutral': probs[:, 1],
                'up': probs[:, 2]
            }
        }

    def save(self, path: str):
        torch.save({
            'model_state': self.model.state_dict(),
            'scaler': self.scaler,
            'sequence_length': self.sequence_length
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.scaler = checkpoint['scaler']
        self.sequence_length = checkpoint['sequence_length']
        # Note: Need to initialize model first with correct input_size
```

#### Day 17: Ensemble Model

**File: `src/models/ensemble.py`**
```python
import numpy as np
from .xgboost_model import XGBoostPredictor
from .lstm_model import LSTMPredictor

class EnsemblePredictor:
    def __init__(self):
        self.xgboost = XGBoostPredictor(use_gpu=True)
        self.lstm = LSTMPredictor(sequence_length=60, use_gpu=True)

        # Weights can be adjusted based on validation performance
        self.weights = {
            'xgboost': 0.5,
            'lstm': 0.5
        }

    def train(self, X: np.array, y: np.array, feature_names: list = None):
        """Train all models"""
        print("=" * 50)
        print("Training XGBoost...")
        print("=" * 50)
        xgb_score = self.xgboost.train(X, y, feature_names)

        print("\n" + "=" * 50)
        print("Training LSTM...")
        print("=" * 50)
        lstm_score = self.lstm.train(X, y)

        # Adjust weights based on performance
        total = xgb_score + lstm_score
        self.weights['xgboost'] = xgb_score / total
        self.weights['lstm'] = lstm_score / total

        print(f"\nFinal weights: XGBoost={self.weights['xgboost']:.2f}, LSTM={self.weights['lstm']:.2f}")

        return {
            'xgboost': xgb_score,
            'lstm': lstm_score,
            'weights': self.weights
        }

    def predict(self, X: np.array) -> dict:
        """Make ensemble prediction"""
        # Get predictions from each model
        xgb_pred = self.xgboost.predict(X)
        lstm_pred = self.lstm.predict(X)

        # Weighted average of probabilities
        ensemble_probs = (
            self.weights['xgboost'] * np.array([
                xgb_pred['probabilities']['down'],
                xgb_pred['probabilities']['neutral'],
                xgb_pred['probabilities']['up']
            ]).T +
            self.weights['lstm'] * np.array([
                lstm_pred['probabilities']['down'],
                lstm_pred['probabilities']['neutral'],
                lstm_pred['probabilities']['up']
            ]).T
        )

        pred_class = np.argmax(ensemble_probs, axis=1)
        confidence = np.max(ensemble_probs, axis=1)

        direction_map = {0: 'down', 1: 'neutral', 2: 'up'}

        return {
            'direction': [direction_map[c] for c in pred_class],
            'confidence': confidence,
            'probabilities': {
                'down': ensemble_probs[:, 0],
                'neutral': ensemble_probs[:, 1],
                'up': ensemble_probs[:, 2]
            },
            'model_predictions': {
                'xgboost': xgb_pred,
                'lstm': lstm_pred
            }
        }

    def save(self, base_path: str):
        self.xgboost.save(f"{base_path}_xgboost.pkl")
        self.lstm.save(f"{base_path}_lstm.pth")
        np.save(f"{base_path}_weights.npy", self.weights)

    def load(self, base_path: str):
        self.xgboost.load(f"{base_path}_xgboost.pkl")
        self.lstm.load(f"{base_path}_lstm.pth")
        self.weights = np.load(f"{base_path}_weights.npy", allow_pickle=True).item()
```

---

### Week 4: Integration & Evaluation

#### Day 18-19: Prediction Pipeline

**File: `src/prediction/predictor.py`**
```python
from ..data.fetcher import DataFetcher
from ..data.database import Database
from ..data.features import FeatureEngineer
from ..patterns.detector import UnifiedPatternDetector
from ..models.ensemble import EnsemblePredictor
import pandas as pd
from datetime import datetime, timedelta

class PredictionPipeline:
    def __init__(self, db_path: str = "data/priceprediction.db"):
        self.db = Database(db_path)
        self.fetcher = DataFetcher()
        self.feature_eng = FeatureEngineer()
        self.pattern_detector = UnifiedPatternDetector()
        self.model = EnsemblePredictor()

    def run_daily_pipeline(self, symbols: list = None):
        """Run full prediction pipeline"""
        symbols = symbols or self.fetcher.default_symbols
        results = []

        for symbol in symbols:
            try:
                result = self._process_symbol(symbol)
                results.append(result)
                print(f"Processed {symbol}: {result['prediction']['direction']} "
                      f"({result['prediction']['confidence']:.2%})")
            except Exception as e:
                print(f"Error processing {symbol}: {e}")

        return results

    def _process_symbol(self, symbol: str) -> dict:
        """Process single symbol"""
        # 1. Fetch latest data
        df = self.fetcher.fetch_history(symbol, period="2y")

        # 2. Create features
        df_features = self.feature_eng.create_features(df)

        # 3. Detect patterns
        patterns = self.pattern_detector.detect_all(df_features)

        # 4. Get features for prediction
        X, _, feature_cols = self.feature_eng.get_feature_matrix(df_features)

        # 5. Make prediction (using last row)
        prediction = self.model.predict(X[-1:])

        # 6. Save prediction to database
        pred_record = {
            'symbol': symbol,
            'prediction_date': datetime.now().date(),
            'target_date': (datetime.now() + timedelta(days=5)).date(),
            'predicted_direction': prediction['direction'][0],
            'predicted_change': None,  # Could add magnitude prediction
            'confidence': float(prediction['confidence'][0]),
            'model_name': 'ensemble_v1'
        }
        self.db.save_prediction(pred_record)

        return {
            'symbol': symbol,
            'prediction': {
                'direction': prediction['direction'][0],
                'confidence': prediction['confidence'][0],
                'probabilities': {k: float(v[0]) for k, v in prediction['probabilities'].items()}
            },
            'patterns': patterns['summary'],
            'model_breakdown': {
                'xgboost': prediction['model_predictions']['xgboost']['direction'],
                'lstm': prediction['model_predictions']['lstm']['direction'][0]
            }
        }

    def train_models(self, symbols: list = None):
        """Train models on historical data"""
        symbols = symbols or self.fetcher.default_symbols

        all_features = []
        all_targets = []

        for symbol in symbols:
            df = self.fetcher.fetch_history(symbol, period="5y")
            df_features = self.feature_eng.create_features(df)
            df_features = self.feature_eng.create_target(df_features, horizon=5)

            # Drop NaN rows
            df_clean = df_features.dropna()

            X, y, feature_cols = self.feature_eng.get_feature_matrix(df_clean)
            all_features.append(X)
            all_targets.append(y)

        X_all = np.vstack(all_features)
        y_all = np.concatenate(all_targets)

        print(f"Training on {len(X_all)} samples with {X_all.shape[1]} features")

        results = self.model.train(X_all, y_all, feature_cols)
        self.model.save("models/ensemble")

        return results
```

**File: `src/prediction/evaluator.py`**
```python
from ..data.database import Database
import pandas as pd
from datetime import datetime, timedelta

class PredictionEvaluator:
    def __init__(self, db_path: str = "data/priceprediction.db"):
        self.db = Database(db_path)

    def update_actuals(self):
        """Update predictions with actual outcomes"""
        # Get predictions that have passed their target date
        query = """
            SELECT id, symbol, target_date, predicted_direction
            FROM predictions
            WHERE actual_direction IS NULL
            AND target_date < date('now')
        """
        pending = pd.read_sql(query, self.db.engine)

        for _, row in pending.iterrows():
            actual = self._get_actual_movement(row['symbol'], row['target_date'])
            if actual:
                self._update_prediction(row['id'], actual)

    def _get_actual_movement(self, symbol: str, target_date) -> dict:
        """Get actual price movement"""
        query = f"""
            SELECT close FROM prices
            WHERE symbol = '{symbol}'
            AND date <= '{target_date}'
            ORDER BY date DESC
            LIMIT 6
        """
        prices = pd.read_sql(query, self.db.engine)

        if len(prices) < 2:
            return None

        start_price = prices.iloc[-1]['close']  # 5 days ago
        end_price = prices.iloc[0]['close']      # target date

        change = (end_price - start_price) / start_price

        if change > 0.02:
            direction = 'up'
        elif change < -0.02:
            direction = 'down'
        else:
            direction = 'neutral'

        return {
            'direction': direction,
            'change': change
        }

    def _update_prediction(self, pred_id: int, actual: dict):
        """Update prediction with actual result"""
        # Get predicted direction
        query = f"SELECT predicted_direction FROM predictions WHERE id = {pred_id}"
        result = pd.read_sql(query, self.db.engine)
        predicted = result.iloc[0]['predicted_direction']

        is_correct = 1 if predicted == actual['direction'] else 0

        update_query = f"""
            UPDATE predictions SET
                actual_direction = '{actual['direction']}',
                actual_change = {actual['change']},
                is_correct = {is_correct}
            WHERE id = {pred_id}
        """
        with self.db.engine.connect() as conn:
            conn.execute(update_query)
            conn.commit()

    def get_accuracy_report(self, days: int = 30) -> dict:
        """Get accuracy report"""
        query = f"""
            SELECT
                model_name,
                predicted_direction,
                COUNT(*) as total,
                SUM(is_correct) as correct,
                AVG(is_correct) * 100 as accuracy
            FROM predictions
            WHERE is_correct IS NOT NULL
            AND prediction_date >= date('now', '-{days} days')
            GROUP BY model_name, predicted_direction
        """
        by_direction = pd.read_sql(query, self.db.engine)

        # Overall accuracy
        overall_query = f"""
            SELECT
                COUNT(*) as total,
                SUM(is_correct) as correct,
                AVG(is_correct) * 100 as accuracy
            FROM predictions
            WHERE is_correct IS NOT NULL
            AND prediction_date >= date('now', '-{days} days')
        """
        overall = pd.read_sql(overall_query, self.db.engine)

        return {
            'overall': overall.to_dict('records')[0] if len(overall) > 0 else {},
            'by_direction': by_direction.to_dict('records'),
            'days_evaluated': days
        }

    def get_confidence_calibration(self) -> pd.DataFrame:
        """Check if confidence scores are well-calibrated"""
        query = """
            SELECT
                ROUND(confidence, 1) as conf_bucket,
                COUNT(*) as total,
                AVG(is_correct) * 100 as actual_accuracy
            FROM predictions
            WHERE is_correct IS NOT NULL
            GROUP BY ROUND(confidence, 1)
            ORDER BY conf_bucket
        """
        return pd.read_sql(query, self.db.engine)
```

#### Day 20-21: FastAPI Backend

**File: `src/api/main.py`**
```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd

from ..prediction.predictor import PredictionPipeline
from ..prediction.evaluator import PredictionEvaluator

app = FastAPI(title="Price Prediction API", version="0.1.0")

# CORS for local React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = PredictionPipeline()
evaluator = PredictionEvaluator()

@app.get("/")
def root():
    return {"status": "running", "version": "0.1.0"}

@app.get("/predictions")
def get_predictions(days: int = 7):
    """Get recent predictions"""
    query = f"""
        SELECT symbol, prediction_date, target_date, predicted_direction,
               confidence, actual_direction, is_correct
        FROM predictions
        WHERE prediction_date >= date('now', '-{days} days')
        ORDER BY prediction_date DESC, confidence DESC
    """
    df = pd.read_sql(query, pipeline.db.engine)
    return df.to_dict('records')

@app.get("/predictions/{symbol}")
def get_symbol_prediction(symbol: str):
    """Get prediction for specific symbol"""
    result = pipeline._process_symbol(symbol.upper())
    return result

@app.post("/predictions/run")
def run_predictions(symbols: Optional[list] = None):
    """Run predictions for all or specified symbols"""
    results = pipeline.run_daily_pipeline(symbols)
    return {"processed": len(results), "results": results}

@app.get("/accuracy")
def get_accuracy(days: int = 30):
    """Get accuracy report"""
    return evaluator.get_accuracy_report(days)

@app.get("/accuracy/calibration")
def get_calibration():
    """Get confidence calibration"""
    df = evaluator.get_confidence_calibration()
    return df.to_dict('records')

@app.post("/train")
def train_models():
    """Trigger model training"""
    results = pipeline.train_models()
    return {"status": "complete", "results": results}

@app.get("/patterns/{symbol}")
def get_patterns(symbol: str):
    """Get detected patterns for symbol"""
    df = pipeline.fetcher.fetch_history(symbol.upper(), period="60d")
    df_features = pipeline.feature_eng.create_features(df)
    patterns = pipeline.pattern_detector.detect_all(df_features)
    return patterns


# Run with: uvicorn src.api.main:app --reload
```

---

### Week 5-6: Minimal UI & Iteration

#### Day 22-24: React Dashboard

**File: `ui/src/App.tsx`**
```tsx
import { useState, useEffect } from 'react'
import './App.css'

interface Prediction {
  symbol: string
  prediction_date: string
  predicted_direction: string
  confidence: number
  actual_direction: string | null
  is_correct: number | null
}

interface Accuracy {
  total: number
  correct: number
  accuracy: number
}

function App() {
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [accuracy, setAccuracy] = useState<Accuracy | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    try {
      const [predRes, accRes] = await Promise.all([
        fetch('http://localhost:8000/predictions?days=14'),
        fetch('http://localhost:8000/accuracy?days=30')
      ])
      const predData = await predRes.json()
      const accData = await accRes.json()

      setPredictions(predData)
      setAccuracy(accData.overall)
    } catch (error) {
      console.error('Error fetching data:', error)
    } finally {
      setLoading(false)
    }
  }

  const runPredictions = async () => {
    setLoading(true)
    await fetch('http://localhost:8000/predictions/run', { method: 'POST' })
    await fetchData()
  }

  const getDirectionColor = (direction: string) => {
    switch (direction) {
      case 'up': return '#22c55e'
      case 'down': return '#ef4444'
      default: return '#9ca3af'
    }
  }

  if (loading) return <div className="loading">Loading...</div>

  return (
    <div className="container">
      <header>
        <h1>Stock Predictions</h1>
        <div className="accuracy-box">
          {accuracy && (
            <>
              <span className="accuracy-value">{accuracy.accuracy?.toFixed(1)}%</span>
              <span className="accuracy-label">Accuracy (30d)</span>
              <span className="accuracy-detail">{accuracy.correct}/{accuracy.total}</span>
            </>
          )}
        </div>
        <button onClick={runPredictions}>Run Predictions</button>
      </header>

      <table>
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Date</th>
            <th>Prediction</th>
            <th>Confidence</th>
            <th>Actual</th>
            <th>Result</th>
          </tr>
        </thead>
        <tbody>
          {predictions.map((pred, i) => (
            <tr key={i}>
              <td className="symbol">{pred.symbol}</td>
              <td>{pred.prediction_date}</td>
              <td style={{ color: getDirectionColor(pred.predicted_direction) }}>
                {pred.predicted_direction.toUpperCase()}
              </td>
              <td>{(pred.confidence * 100).toFixed(0)}%</td>
              <td style={{ color: getDirectionColor(pred.actual_direction || '') }}>
                {pred.actual_direction?.toUpperCase() || '-'}
              </td>
              <td>
                {pred.is_correct === null ? '-' :
                 pred.is_correct === 1 ? '✓' : '✗'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default App
```

**File: `ui/src/App.css`**
```css
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: 'SF Mono', monospace;
  background: #0f172a;
  color: #e2e8f0;
}

.container {
  max-width: 1000px;
  margin: 0 auto;
  padding: 20px;
}

header {
  display: flex;
  align-items: center;
  gap: 20px;
  margin-bottom: 30px;
}

h1 {
  font-size: 24px;
  flex-grow: 1;
}

.accuracy-box {
  display: flex;
  flex-direction: column;
  align-items: center;
  background: #1e293b;
  padding: 10px 20px;
  border-radius: 8px;
}

.accuracy-value {
  font-size: 28px;
  font-weight: bold;
  color: #22c55e;
}

.accuracy-label {
  font-size: 12px;
  color: #94a3b8;
}

.accuracy-detail {
  font-size: 11px;
  color: #64748b;
}

button {
  background: #3b82f6;
  color: white;
  border: none;
  padding: 10px 20px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
}

button:hover {
  background: #2563eb;
}

table {
  width: 100%;
  border-collapse: collapse;
  background: #1e293b;
  border-radius: 8px;
  overflow: hidden;
}

th, td {
  padding: 12px 16px;
  text-align: left;
  border-bottom: 1px solid #334155;
}

th {
  background: #334155;
  font-size: 12px;
  text-transform: uppercase;
  color: #94a3b8;
}

.symbol {
  font-weight: bold;
}

.loading {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  font-size: 18px;
}
```

---

## Scripts

**File: `scripts/train_models.py`**
```python
#!/usr/bin/env python
"""Train all models"""
import sys
sys.path.append('.')

from src.prediction.predictor import PredictionPipeline

if __name__ == "__main__":
    pipeline = PredictionPipeline()
    results = pipeline.train_models()
    print("\nTraining complete!")
    print(f"XGBoost accuracy: {results['xgboost']:.4f}")
    print(f"LSTM accuracy: {results['lstm']:.4f}")
```

**File: `scripts/run_predictions.py`**
```python
#!/usr/bin/env python
"""Run daily predictions"""
import sys
sys.path.append('.')

from src.prediction.predictor import PredictionPipeline
from src.prediction.evaluator import PredictionEvaluator

if __name__ == "__main__":
    # Update actuals first
    evaluator = PredictionEvaluator()
    evaluator.update_actuals()

    # Run new predictions
    pipeline = PredictionPipeline()
    results = pipeline.run_daily_pipeline()

    # Show accuracy
    accuracy = evaluator.get_accuracy_report()
    print(f"\n30-day accuracy: {accuracy['overall'].get('accuracy', 0):.1f}%")
```

---

## Daily Workflow

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Run predictions (do this after market close)
python scripts/run_predictions.py

# 3. Start API
uvicorn src.api.main:app --reload

# 4. View dashboard (in another terminal)
cd ui && npm run dev

# 5. Weekly: Retrain models
python scripts/train_models.py
```

---

## Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Direction Accuracy | >55% | Correct up/down/neutral |
| Confidence Calibration | <10% ECE | Calibration curve |
| Bullish Precision | >60% | When we predict up, how often right |
| Bearish Precision | >60% | When we predict down, how often right |

---

## Iteration Checklist

After 2 weeks of running:

1. [ ] Check accuracy by direction (up vs down vs neutral)
2. [ ] Analyze which features are most important
3. [ ] Identify symbols where model performs poorly
4. [ ] Tune prediction threshold (currently 2%)
5. [ ] Add more features if accuracy is low
6. [ ] Consider adding sentiment data
7. [ ] Test different prediction horizons

---

*Plan Created: 2026-02-02*
*Status: Ready for Implementation*
