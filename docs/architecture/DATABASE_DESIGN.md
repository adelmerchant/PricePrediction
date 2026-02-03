# Database Design & Schema Specification

## Table of Contents
1. [SQL Server Schema](#sql-server-schema)
2. [MongoDB Schema](#mongodb-schema)
3. [Redis Cache Strategy](#redis-cache-strategy)
4. [Data Partitioning Strategy](#data-partitioning-strategy)
5. [Index Design](#index-design)
6. [Data Retention Policies](#data-retention-policies)

---

## SQL Server Schema

### Stocks Table

```sql
CREATE TABLE dbo.Stocks (
    StockId INT IDENTITY(1,1) PRIMARY KEY,
    Symbol NVARCHAR(10) NOT NULL,
    CompanyName NVARCHAR(200) NOT NULL,
    Exchange NVARCHAR(50) NOT NULL,
    Sector NVARCHAR(100),
    Industry NVARCHAR(100),
    MarketCap DECIMAL(18,2),
    Currency NVARCHAR(3) DEFAULT 'USD',
    Country NVARCHAR(50) DEFAULT 'USA',
    IPODate DATE,
    IsActive BIT NOT NULL DEFAULT 1,
    CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    UpdatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),

    CONSTRAINT UK_Stocks_Symbol UNIQUE (Symbol),
    INDEX IX_Stocks_Sector_Industry NONCLUSTERED (Sector, Industry),
    INDEX IX_Stocks_Exchange NONCLUSTERED (Exchange),
    INDEX IX_Stocks_IsActive NONCLUSTERED (IsActive) WHERE IsActive = 1
);
```

---

### OHLCV_Data Table (Partitioned)

```sql
-- Partition Function by Year
CREATE PARTITION FUNCTION PF_OhlcvByYear (DATETIME2)
AS RANGE RIGHT FOR VALUES (
    '2020-01-01', '2021-01-01', '2022-01-01',
    '2023-01-01', '2024-01-01', '2025-01-01', '2026-01-01'
);

-- Partition Scheme
CREATE PARTITION SCHEME PS_OhlcvByYear
AS PARTITION PF_OhlcvByYear
ALL TO ([PRIMARY]);

-- OHLCV Data Table with Columnstore Index
CREATE TABLE dbo.OHLCV_Data (
    OhlcvId BIGINT IDENTITY(1,1),
    StockId INT NOT NULL,
    Timestamp DATETIME2 NOT NULL,
    [Open] DECIMAL(18,4) NOT NULL,
    [High] DECIMAL(18,4) NOT NULL,
    [Low] DECIMAL(18,4) NOT NULL,
    [Close] DECIMAL(18,4) NOT NULL,
    Volume BIGINT NOT NULL,
    VWAP DECIMAL(18,4),
    Trades INT,
    Timeframe TINYINT NOT NULL, -- 1=1min, 5=5min, 60=1hour, 1440=1day
    CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),

    CONSTRAINT PK_OHLCV_Data PRIMARY KEY CLUSTERED (Timestamp, StockId, Timeframe)
        ON PS_OhlcvByYear(Timestamp),
    CONSTRAINT FK_OHLCV_Data_Stocks FOREIGN KEY (StockId) REFERENCES dbo.Stocks(StockId),
    INDEX IX_OHLCV_Stock_Timeframe NONCLUSTERED (StockId, Timeframe, Timestamp),
) ON PS_OhlcvByYear(Timestamp);

-- Columnstore Index for Analytics (10x compression)
CREATE NONCLUSTERED COLUMNSTORE INDEX CSI_OHLCV_Data
ON dbo.OHLCV_Data (StockId, Timestamp, [Open], [High], [Low], [Close], Volume, VWAP, Timeframe);
```

**Sizing Estimates**:
- **1-min data**: ~390 bars/day/stock × 252 days × 3000 stocks = ~295M rows/year
- **Storage**: ~50 bytes/row × 295M = ~14 GB/year (before compression)
- **With Columnstore**: ~1.4 GB/year (90% compression)
- **5 years**: ~7 GB compressed

---

### Predictions Table

```sql
CREATE TABLE dbo.Predictions (
    PredictionId BIGINT IDENTITY(1,1) PRIMARY KEY,
    StockId INT NOT NULL,
    ModelId INT NOT NULL,
    PredictedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    TargetDate DATETIME2 NOT NULL,
    Horizon INT NOT NULL, -- Minutes ahead (5, 60, 1440, etc.)

    -- Prediction Values
    PredictedPrice DECIMAL(18,4) NOT NULL,
    PredictedReturn DECIMAL(10,6),

    -- Confidence Intervals (95%)
    ConfidenceLower DECIMAL(18,4),
    ConfidenceUpper DECIMAL(18,4),
    ConfidenceScore DECIMAL(5,4), -- 0.0 to 1.0

    -- Actual Outcome (filled later)
    ActualPrice DECIMAL(18,4),
    ActualReturn DECIMAL(10,6),
    PredictionError DECIMAL(18,4),
    DirectionCorrect BIT,

    -- Metadata
    Timeframe TINYINT NOT NULL,
    FeatureImportance NVARCHAR(MAX), -- JSON
    ModelVersion NVARCHAR(50),

    CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    UpdatedAt DATETIME2,

    CONSTRAINT FK_Predictions_Stocks FOREIGN KEY (StockId) REFERENCES dbo.Stocks(StockId),
    CONSTRAINT FK_Predictions_Models FOREIGN KEY (ModelId) REFERENCES dbo.ML_Models(ModelId),
    INDEX IX_Predictions_Stock_Target NONCLUSTERED (StockId, TargetDate, Timeframe),
    INDEX IX_Predictions_Model_Date NONCLUSTERED (ModelId, PredictedAt DESC),
    INDEX IX_Predictions_Accuracy NONCLUSTERED (DirectionCorrect, ConfidenceScore)
        WHERE DirectionCorrect IS NOT NULL
);
```

**Sizing Estimates**:
- **3 predictions/day/stock** (intra-day, short, long) × 252 days × 1000 stocks = ~756K rows/year
- **Storage**: ~200 bytes/row × 756K = ~145 MB/year

---

### Alerts Table

```sql
CREATE TABLE dbo.Alerts (
    AlertId BIGINT IDENTITY(1,1) PRIMARY KEY,
    UserId INT NOT NULL,
    StockId INT NOT NULL,

    -- Alert Configuration
    AlertType NVARCHAR(50) NOT NULL, -- 'PriceTarget', 'Pattern', 'Volatility', 'Custom'
    RuleConfig NVARCHAR(MAX) NOT NULL, -- JSON rule definition

    -- Status & Scheduling
    [Status] NVARCHAR(20) NOT NULL DEFAULT 'Active', -- 'Active', 'Triggered', 'Disabled', 'Expired'
    Priority NVARCHAR(10) NOT NULL DEFAULT 'Medium', -- 'High', 'Medium', 'Low'
    RepeatInterval INT, -- Minutes (NULL = one-time alert)
    ExpiresAt DATETIME2,

    -- Trigger Information
    TriggeredAt DATETIME2,
    TriggerValue DECIMAL(18,4),
    TriggerCondition NVARCHAR(MAX), -- JSON snapshot of condition when triggered

    -- Notification
    NotificationChannel NVARCHAR(50) NOT NULL, -- 'Web', 'Email', 'SMS', 'All'
    SentAt DATETIME2,
    DeliveryStatus NVARCHAR(20),

    CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    UpdatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),

    CONSTRAINT FK_Alerts_Users FOREIGN KEY (UserId) REFERENCES dbo.Users(UserId),
    CONSTRAINT FK_Alerts_Stocks FOREIGN KEY (StockId) REFERENCES dbo.Stocks(StockId),
    INDEX IX_Alerts_User_Status NONCLUSTERED (UserId, [Status]) WHERE [Status] = 'Active',
    INDEX IX_Alerts_Stock_Status NONCLUSTERED (StockId, [Status]) WHERE [Status] = 'Active',
    INDEX IX_Alerts_Triggered NONCLUSTERED (TriggeredAt DESC) WHERE TriggeredAt IS NOT NULL
);
```

---

### Patterns Table

```sql
CREATE TABLE dbo.Patterns (
    PatternId BIGINT IDENTITY(1,1) PRIMARY KEY,
    StockId INT NOT NULL,

    -- Pattern Info
    PatternType NVARCHAR(50) NOT NULL, -- 'HeadShoulders', 'DoubleTop', 'Triangle', etc.
    PatternCategory NVARCHAR(20), -- 'Reversal', 'Continuation', 'Consolidation'

    -- Detection
    DetectedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    ConfidenceScore DECIMAL(5,4) NOT NULL, -- 0.0 to 1.0

    -- Time Range
    StartDate DATETIME2 NOT NULL,
    EndDate DATETIME2,
    Timeframe NVARCHAR(10) NOT NULL, -- '1min', '5min', '1hour', '1day'

    -- Pattern Data
    PatternData NVARCHAR(MAX), -- JSON: key levels, dimensions, etc.
    ImagePath NVARCHAR(500), -- Path to pattern chart image

    -- Validation
    IsConfirmed BIT DEFAULT 0,
    ConfirmedAt DATETIME2,
    Outcome NVARCHAR(20), -- 'Success', 'Failure', 'Pending'
    PriceTarget DECIMAL(18,4),
    ActualMove DECIMAL(10,6),

    CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),

    CONSTRAINT FK_Patterns_Stocks FOREIGN KEY (StockId) REFERENCES dbo.Stocks(StockId),
    INDEX IX_Patterns_Stock_Date NONCLUSTERED (StockId, DetectedAt DESC),
    INDEX IX_Patterns_Type_Confidence NONCLUSTERED (PatternType, ConfidenceScore DESC),
    INDEX IX_Patterns_Pending NONCLUSTERED (StockId, IsConfirmed) WHERE IsConfirmed = 0
);
```

---

### ML_Models Table

```sql
CREATE TABLE dbo.ML_Models (
    ModelId INT IDENTITY(1,1) PRIMARY KEY,
    ModelName NVARCHAR(100) NOT NULL,
    ModelType NVARCHAR(50) NOT NULL, -- 'LSTM', 'GRU', 'Transformer', 'XGBoost', etc.
    ModelCategory NVARCHAR(20) NOT NULL, -- 'Intraday', 'ShortTerm', 'LongTerm'
    Version NVARCHAR(50) NOT NULL,

    -- Training Configuration
    Hyperparameters NVARCHAR(MAX), -- JSON
    TrainingConfig NVARCHAR(MAX), -- JSON: epochs, batch size, etc.

    -- Performance Metrics (Validation Set)
    ValidationAccuracy DECIMAL(10,6),
    ValidationMAE DECIMAL(18,6),
    ValidationRMSE DECIMAL(18,6),
    ValidationMAPE DECIMAL(10,6),

    -- Performance Metrics (Test Set)
    TestAccuracy DECIMAL(10,6),
    TestMAE DECIMAL(18,6),
    TestRMSE DECIMAL(18,6),
    TestMAPE DECIMAL(10,6),
    TestSharpe DECIMAL(10,4),

    -- Training Details
    TrainingStarted DATETIME2,
    TrainingCompleted DATETIME2,
    TrainingDurationMinutes INT,
    EpochsCompleted INT,

    -- Storage
    StoragePath NVARCHAR(500), -- Path to ONNX model file
    ModelSizeBytes BIGINT,

    -- Status & Deployment
    [Status] NVARCHAR(20) NOT NULL DEFAULT 'Training', -- 'Training', 'Validating', 'Deployed', 'Archived'
    IsProduction BIT NOT NULL DEFAULT 0,
    DeployedAt DATETIME2,
    ArchivedAt DATETIME2,

    CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    UpdatedAt DATETIME2,

    CONSTRAINT UK_ML_Models_Name_Version UNIQUE (ModelName, Version),
    INDEX IX_ML_Models_Type_Status NONCLUSTERED (ModelType, [Status]),
    INDEX IX_ML_Models_Production NONCLUSTERED (IsProduction, ModelCategory) WHERE IsProduction = 1,
    INDEX IX_ML_Models_Performance NONCLUSTERED (ModelType, TestAccuracy DESC)
);
```

---

### Backtests Table

```sql
CREATE TABLE dbo.Backtests (
    BacktestId BIGINT IDENTITY(1,1) PRIMARY KEY,
    ModelId INT NOT NULL,
    BacktestName NVARCHAR(200) NOT NULL,

    -- Backtest Configuration
    StartDate DATE NOT NULL,
    EndDate DATE NOT NULL,
    InitialCapital DECIMAL(18,2) NOT NULL,
    Configuration NVARCHAR(MAX), -- JSON: position sizing, risk params, etc.

    -- Performance Summary
    TotalReturn DECIMAL(10,4),
    AnnualizedReturn DECIMAL(10,4),
    TotalTrades INT,
    WinningTrades INT,
    LosingTrades INT,
    WinRate DECIMAL(5,4),

    -- Risk Metrics
    SharpeRatio DECIMAL(10,4),
    SortinoRatio DECIMAL(10,4),
    CalmarRatio DECIMAL(10,4),
    MaxDrawdown DECIMAL(10,4),
    MaxDrawdownDuration INT, -- Days

    -- Trade Metrics
    AverageWin DECIMAL(18,4),
    AverageLoss DECIMAL(18,4),
    ProfitFactor DECIMAL(10,4),
    AverageTradeDuration DECIMAL(10,2), -- Hours

    -- Comparison to Benchmark
    BenchmarkReturn DECIMAL(10,4), -- S&P 500
    Alpha DECIMAL(10,4),
    Beta DECIMAL(10,4),

    -- Full Metrics
    PerformanceMetrics NVARCHAR(MAX), -- JSON: detailed metrics, equity curve

    -- Status
    [Status] NVARCHAR(20) NOT NULL DEFAULT 'Running', -- 'Running', 'Completed', 'Failed'
    CompletedAt DATETIME2,
    ErrorMessage NVARCHAR(MAX),

    CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),

    CONSTRAINT FK_Backtests_Models FOREIGN KEY (ModelId) REFERENCES dbo.ML_Models(ModelId),
    INDEX IX_Backtests_Model NONCLUSTERED (ModelId, CreatedAt DESC),
    INDEX IX_Backtests_Performance NONCLUSTERED (SharpeRatio DESC, MaxDrawdown ASC)
        WHERE [Status] = 'Completed'
);
```

---

### Backtest_Trades Table (Partitioned)

```sql
CREATE TABLE dbo.Backtest_Trades (
    TradeId BIGINT IDENTITY(1,1),
    BacktestId BIGINT NOT NULL,
    StockId INT NOT NULL,

    -- Entry
    EntryDate DATETIME2 NOT NULL,
    EntryPrice DECIMAL(18,4) NOT NULL,
    EntrySignal NVARCHAR(MAX), -- JSON: why entered

    -- Exit
    ExitDate DATETIME2,
    ExitPrice DECIMAL(18,4),
    ExitSignal NVARCHAR(MAX), -- JSON: why exited

    -- Trade Details
    Quantity INT NOT NULL,
    TradeType NVARCHAR(10) NOT NULL, -- 'Long', 'Short'
    [Status] NVARCHAR(20) NOT NULL DEFAULT 'Open', -- 'Open', 'Closed'

    -- P&L
    GrossProfitLoss DECIMAL(18,4),
    Commission DECIMAL(18,4) DEFAULT 0,
    Slippage DECIMAL(18,4) DEFAULT 0,
    NetProfitLoss DECIMAL(18,4),
    ReturnPercent DECIMAL(10,6),

    -- Duration
    DurationMinutes INT,

    -- Risk
    StopLoss DECIMAL(18,4),
    TakeProfit DECIMAL(18,4),
    MaxAdverseExcursion DECIMAL(10,6), -- MAE
    MaxFavorableExcursion DECIMAL(10,6), -- MFE

    CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    UpdatedAt DATETIME2,

    CONSTRAINT PK_Backtest_Trades PRIMARY KEY CLUSTERED (TradeId, BacktestId),
    CONSTRAINT FK_Backtest_Trades_Backtests FOREIGN KEY (BacktestId) REFERENCES dbo.Backtests(BacktestId) ON DELETE CASCADE,
    CONSTRAINT FK_Backtest_Trades_Stocks FOREIGN KEY (StockId) REFERENCES dbo.Stocks(StockId),
    INDEX IX_Backtest_Trades_Stock NONCLUSTERED (StockId, EntryDate DESC),
    INDEX IX_Backtest_Trades_Performance NONCLUSTERED (BacktestId, NetProfitLoss DESC)
);
```

---

### Fundamentals Table

```sql
CREATE TABLE dbo.Fundamentals (
    FundamentalId BIGINT IDENTITY(1,1) PRIMARY KEY,
    StockId INT NOT NULL,
    ReportDate DATE NOT NULL,
    Period NVARCHAR(10) NOT NULL, -- 'Q1', 'Q2', 'Q3', 'Q4', 'FY'
    FiscalYear INT NOT NULL,

    -- Income Statement
    Revenue DECIMAL(20,2),
    CostOfRevenue DECIMAL(20,2),
    GrossProfit DECIMAL(20,2),
    OperatingIncome DECIMAL(20,2),
    NetIncome DECIMAL(20,2),
    EPS DECIMAL(10,4),
    DilutedEPS DECIMAL(10,4),

    -- Balance Sheet
    TotalAssets DECIMAL(20,2),
    TotalLiabilities DECIMAL(20,2),
    TotalEquity DECIMAL(20,2),
    Cash DECIMAL(20,2),
    TotalDebt DECIMAL(20,2),

    -- Cash Flow
    OperatingCashFlow DECIMAL(20,2),
    CapitalExpenditures DECIMAL(20,2),
    FreeCashFlow DECIMAL(20,2),

    -- Ratios
    PE_Ratio DECIMAL(10,4),
    PB_Ratio DECIMAL(10,4),
    PS_Ratio DECIMAL(10,4),
    PEG_Ratio DECIMAL(10,4),
    ROE DECIMAL(10,6),
    ROA DECIMAL(10,6),
    DebtToEquity DECIMAL(10,4),
    CurrentRatio DECIMAL(10,4),
    QuickRatio DECIMAL(10,4),

    -- Growth Metrics
    RevenueGrowthYoY DECIMAL(10,6),
    EPSGrowthYoY DECIMAL(10,6),

    CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    UpdatedAt DATETIME2,

    CONSTRAINT FK_Fundamentals_Stocks FOREIGN KEY (StockId) REFERENCES dbo.Stocks(StockId),
    CONSTRAINT UK_Fundamentals_Stock_Period UNIQUE (StockId, ReportDate, Period),
    INDEX IX_Fundamentals_Stock_Date NONCLUSTERED (StockId, ReportDate DESC)
);
```

---

### Users Table (Single User, but extensible)

```sql
CREATE TABLE dbo.Users (
    UserId INT IDENTITY(1,1) PRIMARY KEY,
    Email NVARCHAR(255) NOT NULL,
    PasswordHash NVARCHAR(255), -- NULL for local development (no auth)
    FirstName NVARCHAR(100),
    LastName NVARCHAR(100),

    -- Preferences
    Preferences NVARCHAR(MAX), -- JSON: theme, default timeframe, notifications, etc.

    -- Session
    LastLoginAt DATETIME2,
    LoginCount INT DEFAULT 0,

    CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    UpdatedAt DATETIME2,

    CONSTRAINT UK_Users_Email UNIQUE (Email)
);
```

---

### Watchlists & Watchlist_Items

```sql
CREATE TABLE dbo.Watchlists (
    WatchlistId INT IDENTITY(1,1) PRIMARY KEY,
    UserId INT NOT NULL,
    Name NVARCHAR(100) NOT NULL,
    Description NVARCHAR(500),
    IsDefault BIT NOT NULL DEFAULT 0,
    SortOrder INT DEFAULT 0,

    CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    UpdatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),

    CONSTRAINT FK_Watchlists_Users FOREIGN KEY (UserId) REFERENCES dbo.Users(UserId),
    INDEX IX_Watchlists_User NONCLUSTERED (UserId, SortOrder)
);

CREATE TABLE dbo.Watchlist_Items (
    WatchlistItemId INT IDENTITY(1,1) PRIMARY KEY,
    WatchlistId INT NOT NULL,
    StockId INT NOT NULL,
    SortOrder INT DEFAULT 0,
    CustomSettings NVARCHAR(MAX), -- JSON: custom alerts, notes, etc.

    AddedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),

    CONSTRAINT FK_WatchlistItems_Watchlists FOREIGN KEY (WatchlistId) REFERENCES dbo.Watchlists(WatchlistId) ON DELETE CASCADE,
    CONSTRAINT FK_WatchlistItems_Stocks FOREIGN KEY (StockId) REFERENCES dbo.Stocks(StockId),
    CONSTRAINT UK_WatchlistItems_Watchlist_Stock UNIQUE (WatchlistId, StockId),
    INDEX IX_WatchlistItems_Watchlist NONCLUSTERED (WatchlistId, SortOrder)
);
```

---

## MongoDB Schema

### Collection: raw_market_data

**Purpose**: Store raw API responses for audit trail

```javascript
{
  _id: ObjectId("..."),
  symbol: "AAPL",
  timestamp: ISODate("2026-02-03T14:30:00Z"),
  source: "polygon.io",
  api_endpoint: "/v2/aggs/ticker/AAPL/range/1/minute/...",
  request_params: {
    timespan: "minute",
    multiplier: 1,
    from: "2026-02-03",
    to: "2026-02-03"
  },
  raw_response: {
    // Full JSON response from API
    ticker: "AAPL",
    results: [...],
    status: "OK",
    request_id: "..."
  },
  response_code: 200,
  response_time_ms: 145,
  metadata: {
    data_points: 390,
    api_credits_used: 1
  },
  created_at: ISODate("2026-02-03T14:30:05Z"),
  ttl_expires_at: ISODate("2026-03-03T14:30:05Z") // 30-day retention
}

// Indexes
db.raw_market_data.createIndex({ symbol: 1, timestamp: -1 });
db.raw_market_data.createIndex({ source: 1, timestamp: -1 });
db.raw_market_data.createIndex({ ttl_expires_at: 1 }, { expireAfterSeconds: 0 }); // TTL index
```

---

### Collection: news_articles

**Purpose**: Financial news with sentiment analysis

```javascript
{
  _id: ObjectId("..."),
  headline: "Apple announces record quarterly earnings",
  summary: "Apple Inc. reported...",
  content: "Full article text...",
  source: "newsapi.org",
  author: "Jane Doe",
  published_at: ISODate("2026-02-03T09:00:00Z"),
  url: "https://...",

  // Symbols mentioned
  symbols: ["AAPL", "MSFT"],
  primary_symbol: "AAPL",

  // Sentiment Analysis
  sentiment_score: 0.85, // -1 to 1 (negative to positive)
  sentiment_label: "positive", // "positive", "negative", "neutral"
  sentiment_magnitude: 0.9, // 0 to 1 (strength)

  // Named Entity Recognition
  entities: [
    { text: "Tim Cook", type: "PERSON", relevance: 0.9 },
    { text: "iPhone", type: "PRODUCT", relevance: 0.8 },
    { text: "Q4 2025", type: "DATE", relevance: 0.7 }
  ],

  // Keywords
  keywords: ["earnings", "revenue", "guidance", "iPhone"],

  // Engagement
  social_shares: 1250,
  comments_count: 45,

  created_at: ISODate("2026-02-03T09:05:00Z"),
  ttl_expires_at: ISODate("2026-05-03T09:05:00Z") // 90-day retention
}

// Indexes
db.news_articles.createIndex({ symbols: 1, published_at: -1 });
db.news_articles.createIndex({ primary_symbol: 1, sentiment_score: -1 });
db.news_articles.createIndex({ published_at: -1 });
db.news_articles.createIndex({ ttl_expires_at: 1 }, { expireAfterSeconds: 0 });
db.news_articles.createIndex({ "$**": "text" }); // Full-text search
```

---

### Collection: model_artifacts

**Purpose**: Store ML model metadata and binaries

```javascript
{
  _id: ObjectId("..."),
  model_id: 42, // References SQL ML_Models.ModelId
  model_name: "LSTM_Intraday_v3",
  model_type: "LSTM",
  version: "3.2.1",

  // Training Details
  hyperparameters: {
    hidden_size: 128,
    num_layers: 3,
    dropout: 0.2,
    learning_rate: 0.001,
    batch_size: 64,
    epochs: 100,
    sequence_length: 60
  },

  training_config: {
    optimizer: "Adam",
    loss_function: "MSE",
    early_stopping_patience: 10,
    validation_split: 0.15
  },

  // Training History
  training_history: {
    epochs: [
      { epoch: 1, train_loss: 0.045, val_loss: 0.052, val_mae: 0.023 },
      { epoch: 2, train_loss: 0.038, val_loss: 0.048, val_mae: 0.021 },
      // ...
      { epoch: 87, train_loss: 0.012, val_loss: 0.015, val_mae: 0.008 }
    ],
    best_epoch: 87,
    early_stopped: true
  },

  // Feature Information
  feature_columns: [
    "close_price_norm",
    "volume_norm",
    "rsi_14",
    "macd",
    "bollinger_upper",
    "bollinger_lower",
    // ... 50 features
  ],
  feature_importance: {
    "close_price_norm": 0.45,
    "volume_norm": 0.12,
    "rsi_14": 0.08,
    // ...
  },

  // Model Binary (GridFS for large files)
  model_binary_gridfs_id: ObjectId("..."), // Reference to GridFS file
  model_size_bytes: 45678912,
  onnx_opset_version: 14,

  // Performance on Test Set
  test_metrics: {
    mae: 0.0082,
    rmse: 0.0154,
    mape: 1.23,
    directional_accuracy: 0.67,
    sharpe_ratio: 1.45
  },

  // Deployment Info
  is_production: true,
  deployed_at: ISODate("2026-02-01T10:00:00Z"),
  inference_latency_ms: {
    p50: 12,
    p95: 18,
    p99: 25
  },

  created_at: ISODate("2026-01-28T15:30:00Z"),
  updated_at: ISODate("2026-02-01T10:00:00Z")
}

// Indexes
db.model_artifacts.createIndex({ model_id: 1 }, { unique: true });
db.model_artifacts.createIndex({ model_name: 1, version: 1 }, { unique: true });
db.model_artifacts.createIndex({ is_production: 1, model_type: 1 });
```

---

### Collection: feature_store

**Purpose**: Pre-computed features for ML inference

```javascript
{
  _id: ObjectId("..."),
  stock_id: 123,
  symbol: "AAPL",
  timestamp: ISODate("2026-02-03T14:30:00Z"),
  timeframe: "1min",

  // Technical Indicators
  features: {
    technical: {
      // Price-based
      close_price: 182.45,
      returns_1min: 0.0012,
      returns_5min: 0.0045,
      volatility_20: 0.023,

      // Moving Averages
      sma_10: 182.30,
      sma_20: 181.85,
      ema_12: 182.40,
      ema_26: 181.90,
      vwap: 182.15,

      // Oscillators
      rsi_14: 58.5,
      stochastic_k: 62.3,
      stochastic_d: 59.1,
      williams_r: -37.7,
      cci_20: 45.2,

      // MACD
      macd: 0.50,
      macd_signal: 0.45,
      macd_histogram: 0.05,

      // Bollinger Bands
      bb_upper: 183.20,
      bb_middle: 182.00,
      bb_lower: 180.80,
      bb_width: 2.40,
      bb_percent: 0.63,

      // Volume
      volume: 1250000,
      volume_sma_20: 1180000,
      volume_ratio: 1.06,
      obv: 45678901234,

      // ATR
      atr_14: 1.85,
      atr_percent: 1.01,

      // ADX
      adx_14: 28.5,
      plus_di: 22.3,
      minus_di: 18.7
    },

    fundamental: {
      pe_ratio: 28.5,
      pb_ratio: 12.3,
      market_cap: 2850000000000,
      eps_ttm: 6.42
    },

    sentiment: {
      news_sentiment_24h: 0.65,
      news_count_24h: 12,
      social_sentiment: 0.58
    },

    macro: {
      spy_return: 0.0015,
      vix: 18.5,
      sector_return: 0.0020
    }
  },

  // Normalized features (ready for ML)
  features_normalized: {
    // StandardScaler normalized values
    // Same keys as above
  },

  computed_at: ISODate("2026-02-03T14:30:05Z"),
  ttl_expires_at: ISODate("2026-02-10T14:30:05Z") // 7-day retention
}

// Indexes
db.feature_store.createIndex({ stock_id: 1, timestamp: -1, timeframe: 1 });
db.feature_store.createIndex({ symbol: 1, timestamp: -1 });
db.feature_store.createIndex({ ttl_expires_at: 1 }, { expireAfterSeconds: 0 });
```

---

### Collection: pattern_instances

**Purpose**: Detected chart patterns with metadata

```javascript
{
  _id: ObjectId("..."),
  pattern_id: 5678, // References SQL Patterns.PatternId
  stock_id: 123,
  symbol: "AAPL",
  pattern_type: "HeadAndShoulders",
  pattern_category: "Reversal",

  // Detection
  detected_at: ISODate("2026-02-03T10:00:00Z"),
  confidence_score: 0.87,

  // Pattern Geometry
  pattern_geometry: {
    left_shoulder: {
      date: ISODate("2026-01-15T00:00:00Z"),
      price: 178.50
    },
    head: {
      date: ISODate("2026-01-25T00:00:00Z"),
      price: 185.20
    },
    right_shoulder: {
      date: ISODate("2026-02-02T00:00:00Z"),
      price: 179.10
    },
    neckline: {
      slope: -0.05,
      intercept: 175.80
    }
  },

  // Key Levels
  support_level: 175.50,
  resistance_level: 185.50,
  price_target: 172.00, // Measured move

  // Volume Profile
  volume_profile: {
    left_shoulder_volume: 85000000,
    head_volume: 120000000,
    right_shoulder_volume: 78000000,
    volume_confirmation: true
  },

  // Chart Image (GridFS reference)
  chart_image_gridfs_id: ObjectId("..."),
  chart_image_url: "/api/patterns/5678/chart.png",

  // Pattern State
  is_confirmed: false,
  confirmation_date: null,
  outcome: "pending", // "success", "failure", "pending"
  actual_move_percent: null,

  // Time Range
  start_date: ISODate("2026-01-10T00:00:00Z"),
  end_date: ISODate("2026-02-03T00:00:00Z"),
  timeframe: "1day",

  created_at: ISODate("2026-02-03T10:00:00Z"),
  updated_at: ISODate("2026-02-03T10:00:00Z")
}

// Indexes
db.pattern_instances.createIndex({ pattern_id: 1 }, { unique: true });
db.pattern_instances.createIndex({ stock_id: 1, detected_at: -1 });
db.pattern_instances.createIndex({ symbol: 1, pattern_type: 1, is_confirmed: 1 });
db.pattern_instances.createIndex({ confidence_score: -1, is_confirmed: 1 });
```

---

## Redis Cache Strategy

### Key Naming Conventions

```
{namespace}:{entity}:{id}:{field}

Examples:
- quote:AAPL                      # Latest quote for AAPL
- quote:AAPL:1min                 # Latest 1-min bar
- prediction:AAPL:intraday        # Intraday prediction for AAPL
- stock:meta:AAPL                 # Stock metadata
- model:lstm_v3:metadata          # Model metadata
- alerts:user:1:active            # Active alerts for user 1
```

### Data Structure Design

#### 1. Latest Quotes (String)

```redis
Key: quote:AAPL
TTL: 300 seconds (5 minutes)
Value: {
  "symbol": "AAPL",
  "price": 182.45,
  "change": 1.25,
  "changePercent": 0.69,
  "volume": 52000000,
  "timestamp": "2026-02-03T14:30:00Z",
  "bid": 182.44,
  "ask": 182.46,
  "high": 183.50,
  "low": 180.20,
  "open": 181.20,
  "prevClose": 181.20
}
```

#### 2. Stock Metadata (Hash)

```redis
Key: stock:meta:AAPL
TTL: 86400 seconds (24 hours)
Fields:
  symbol: "AAPL"
  name: "Apple Inc."
  sector: "Technology"
  industry: "Consumer Electronics"
  marketCap: "2850000000000"
  exchange: "NASDAQ"
```

#### 3. Predictions (String with JSON)

```redis
Key: prediction:AAPL:intraday
TTL: 3600 seconds (1 hour)
Value: {
  "symbol": "AAPL",
  "timeframe": "intraday",
  "predictedPrice": 183.20,
  "confidenceLower": 182.50,
  "confidenceUpper": 183.90,
  "confidenceScore": 0.78,
  "targetDate": "2026-02-03T15:30:00Z",
  "predictedAt": "2026-02-03T14:30:00Z",
  "modelVersion": "lstm_v3"
}
```

#### 4. Top Movers (Sorted Set)

```redis
Key: market:top_gainers
TTL: 300 seconds (5 minutes)
Score: percent_change
Members: symbol

ZADD market:top_gainers 5.25 "NVDA"
ZADD market:top_gainers 3.80 "TSLA"
ZADD market:top_gainers 2.45 "AAPL"

# Get top 10 gainers
ZREVRANGE market:top_gainers 0 9 WITHSCORES
```

#### 5. Recent Alerts (List)

```redis
Key: alerts:recent
TTL: 86400 seconds (24 hours)
Max Length: 100 (LTRIM to keep last 100)

LPUSH alerts:recent '{"symbol":"AAPL","type":"PriceTarget","message":"...","timestamp":"..."}'
LTRIM alerts:recent 0 99
```

#### 6. Real-time Quote Stream (Redis Streams)

```redis
Stream: quotes:stream

XADD quotes:stream * symbol AAPL price 182.45 volume 1250000 timestamp 1738592400

# Consumer groups for different services
XGROUP CREATE quotes:stream prediction_service $ MKSTREAM
XGROUP CREATE quotes:stream alert_service $ MKSTREAM
```

#### 7. Model Metadata Cache (Hash)

```redis
Key: model:lstm_v3:metadata
TTL: 3600 seconds (1 hour)
Fields:
  modelId: "42"
  modelName: "LSTM_Intraday_v3"
  version: "3.2.1"
  accuracy: "0.67"
  isProduction: "true"
  deployedAt: "2026-02-01T10:00:00Z"
```

#### 8. Feature Vectors (String)

```redis
Key: features:AAPL:1min:latest
TTL: 600 seconds (10 minutes)
Value: {
  "stockId": 123,
  "timestamp": "2026-02-03T14:30:00Z",
  "features": [0.023, 0.58, -0.12, ...], // Normalized feature vector
  "featureNames": ["close_norm", "volume_norm", "rsi_14", ...]
}
```

### Pub/Sub Channels

```redis
# Real-time quote updates
PUBLISH quotes:channel '{"symbol":"AAPL","price":182.45,"timestamp":"..."}'

# Alert notifications
PUBLISH alerts:channel '{"userId":1,"symbol":"AAPL","type":"PriceTarget","message":"..."}'

# Model updates
PUBLISH models:channel '{"modelId":42,"event":"deployed","version":"3.2.1"}'

# Pattern detection
PUBLISH patterns:channel '{"symbol":"AAPL","pattern":"HeadAndShoulders","confidence":0.87}'
```

---

## Data Partitioning Strategy

### SQL Server Partitioning

**OHLCV_Data Table**: Partition by **Timestamp** (Year boundaries)

**Benefits**:
- Query performance: Most queries filter by date range
- Maintenance: Easier to archive/drop old partitions
- Load performance: Parallel loading into different partitions

**Partition Scheme**:
```
Partition 1: < 2020-01-01
Partition 2: 2020-01-01 to 2020-12-31
Partition 3: 2021-01-01 to 2021-12-31
...
Partition 8: 2026-01-01 to 2026-12-31
Partition 9: >= 2027-01-01
```

**Monthly Partitioning** (for very high volume):
```sql
-- For intra-day 1-min data, consider monthly partitions
CREATE PARTITION FUNCTION PF_OhlcvByMonth (DATETIME2)
AS RANGE RIGHT FOR VALUES (
    '2026-01-01', '2026-02-01', '2026-03-01',
    '2026-04-01', '2026-05-01', '2026-06-01',
    -- etc...
);
```

---

### MongoDB Sharding (Future)

**Shard Key**: `{ symbol: 1, timestamp: -1 }`

**Rationale**:
- Even distribution across stocks
- Range queries on timestamp are efficient
- Co-location of data for same symbol

**Sharding Threshold**: Shard when collection exceeds 500 GB

---

## Index Design

### SQL Server Indexing Strategy

#### Covering Indexes for Common Queries

```sql
-- Query: Get recent OHLCV data for a stock
CREATE NONCLUSTERED INDEX IX_OHLCV_Stock_Timeframe_Covering
ON dbo.OHLCV_Data (StockId, Timeframe, Timestamp DESC)
INCLUDE ([Open], [High], [Low], [Close], Volume, VWAP);

-- Query: Get all predictions for a stock with model info
CREATE NONCLUSTERED INDEX IX_Predictions_Stock_Covering
ON dbo.Predictions (StockId, TargetDate DESC)
INCLUDE (ModelId, PredictedPrice, ConfidenceScore, Timeframe);

-- Query: Get active alerts for a user
CREATE NONCLUSTERED INDEX IX_Alerts_User_Active_Covering
ON dbo.Alerts (UserId, [Status])
INCLUDE (StockId, AlertType, TriggeredAt, Priority)
WHERE [Status] = 'Active';
```

#### Filtered Indexes

```sql
-- Index only active stocks
CREATE NONCLUSTERED INDEX IX_Stocks_Active
ON dbo.Stocks (Symbol, Sector, Industry)
WHERE IsActive = 1;

-- Index only production models
CREATE NONCLUSTERED INDEX IX_Models_Production
ON dbo.ML_Models (ModelType, TestAccuracy DESC)
WHERE IsProduction = 1;
```

---

### MongoDB Indexing Strategy

```javascript
// Compound index for time-series queries
db.raw_market_data.createIndex({ symbol: 1, timestamp: -1 });

// Compound index for news sentiment analysis
db.news_articles.createIndex({ primary_symbol: 1, sentiment_score: -1, published_at: -1 });

// Text index for full-text search on news
db.news_articles.createIndex({
  headline: "text",
  content: "text",
  keywords: "text"
}, {
  weights: {
    headline: 10,
    keywords: 5,
    content: 1
  }
});

// TTL index for automatic data expiration
db.raw_market_data.createIndex({ ttl_expires_at: 1 }, { expireAfterSeconds: 0 });
db.news_articles.createIndex({ ttl_expires_at: 1 }, { expireAfterSeconds: 0 });
db.feature_store.createIndex({ ttl_expires_at: 1 }, { expireAfterSeconds: 0 });
```

---

## Data Retention Policies

### SQL Server Retention

| Table | Retention | Archive Strategy |
|-------|-----------|------------------|
| OHLCV_Data (1-min) | 1 year | Archive to cold storage after 1 year, keep 5-min aggregates |
| OHLCV_Data (5-min) | 3 years | Archive to cold storage after 3 years |
| OHLCV_Data (1-day) | Indefinite | Keep forever (compact) |
| Predictions | 2 years | Delete or archive after 2 years |
| Patterns | Indefinite | Keep confirmed patterns forever |
| Alerts | 1 year | Delete after 1 year |
| Backtest_Trades | Indefinite | Keep for analysis |
| Fundamentals | Indefinite | Historical fundamentals are valuable |

### MongoDB Retention (TTL)

| Collection | TTL | Reasoning |
|------------|-----|-----------|
| raw_market_data | 30 days | Audit trail, data already in SQL |
| news_articles | 90 days | Sentiment loses relevance |
| feature_store | 7 days | Recomputed on demand |
| model_artifacts | Indefinite | Keep all model versions |
| pattern_instances | Indefinite | Historical patterns are valuable |

### Redis Retention (TTL)

| Key Pattern | TTL | Reasoning |
|-------------|-----|-----------|
| quote:* | 5 minutes | Real-time quotes stale quickly |
| prediction:* | 1 hour | Predictions updated hourly |
| stock:meta:* | 24 hours | Metadata changes infrequently |
| model:*:metadata | 1 hour | Model metadata semi-static |
| alerts:recent | 24 hours | Recent alerts for quick access |
| features:*:latest | 10 minutes | Features recomputed frequently |

---

**Document Version**: 1.0
**Last Updated**: 2026-02-03
