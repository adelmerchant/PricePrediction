# US Stock Market Price Prediction Application - Architecture Overview

## Executive Summary

This document outlines the architecture for a single-user, GPU-accelerated stock market price prediction application running primarily on a local workstation with NVIDIA RTX 3090 GPU. The system leverages machine learning for pattern detection, trend analysis, and price prediction across multiple timeframes (intra-day, short-term, long-term).

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Core Services](#core-services)
3. [Technology Stack](#technology-stack)
4. [Data Architecture](#data-architecture)
5. [Analysis Timeframes](#analysis-timeframes)
6. [Deployment Architecture](#deployment-architecture)

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Presentation Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │  Web UI      │  │  REST API    │  │  SignalR Hub       │    │
│  │  (Blazor)    │  │  (.NET 10)   │  │  (Real-time)       │    │
│  └──────────────┘  └──────────────┘  └────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      Application Services Layer                  │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ Trend        │  │ Pattern      │  │ Price Prediction   │    │
│  │ Detection    │  │ Detection    │  │ Service            │    │
│  └──────────────┘  └──────────────┘  └────────────────────┘    │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ Alert        │  │ Backtesting  │  │ Portfolio          │    │
│  │ Processor    │  │ Engine       │  │ Analytics          │    │
│  └──────────────┘  └──────────────┘  └────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                     ML & Analytics Engine                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  ML Pipeline (CUDA-Accelerated)                          │   │
│  │  ┌────────────┐ ┌────────────┐ ┌──────────────────────┐ │   │
│  │  │ LSTM/GRU   │ │ Transformer│ │ Ensemble Models      │ │   │
│  │  │ Networks   │ │ Models     │ │ (XGBoost, LightGBM)  │ │   │
│  │  └────────────┘ └────────────┘ └──────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Feature Engineering & Technical Indicators              │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      Data Access Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ SQL Server   │  │ MongoDB      │  │ Redis Cache        │    │
│  │ Repository   │  │ Repository   │  │ Service            │    │
│  └──────────────┘  └──────────────┘  └────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      Background Services                         │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ Data Fetcher │  │ Model        │  │ Alert Monitor      │    │
│  │ Service      │  │ Trainer      │  │ Service            │    │
│  │ (Quartz.NET) │  │ Service      │  │ (Hangfire)         │    │
│  └──────────────┘  └──────────────┘  └────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      External Data Sources                       │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ Polygon.io   │  │ Alpha Vantage│  │ Yahoo Finance      │    │
│  │ (Primary)    │  │ (Backup)     │  │ (Free/Backup)      │    │
│  └──────────────┘  └──────────────┘  └────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Services

### 1. Data Fetcher Service
**Purpose**: Acquire real-time and historical market data from external sources

**Responsibilities**:
- Fetch real-time stock quotes (1-min, 5-min intervals for intra-day)
- Download historical OHLCV data
- Retrieve fundamental data (earnings, ratios, news)
- Handle API rate limiting and failover
- Data validation and cleansing
- Store raw data in MongoDB for audit trail

**Technology**:
- **Background Scheduler**: Quartz.NET for scheduling data fetches
- **HTTP Client**: Polly for retry policies and circuit breakers
- **Data Sources Integration**: Multiple provider adapters (Strategy pattern)

**Scheduling**:
- Real-time quotes: Every 1-5 minutes during market hours
- Historical data: Daily after market close
- Fundamental data: Weekly
- News sentiment: Hourly

---

### 2. Machine Learning Service
**Purpose**: Train, validate, and serve ML models for price prediction

**Responsibilities**:
- Feature engineering from raw market data
- Model training with GPU acceleration
- Model versioning and A/B testing
- Hyperparameter optimization
- Model performance monitoring
- Inference serving

**Technology Stack**:
- **ML Framework**:
  - **ML.NET** (native .NET integration) - Primary for traditional ML
  - **TorchSharp** (PyTorch bindings for .NET) - For deep learning with CUDA support
  - **ONNX Runtime** - For optimized inference with GPU acceleration
- **GPU Acceleration**: CUDA 12.x, cuDNN 8.x
- **Model Storage**: MongoDB (model metadata), File System (model binaries)
- **Experiment Tracking**: MLflow (can run locally)

**Model Architecture**:
1. **Intra-day Models**:
   - LSTM with attention mechanism (1-min to 1-hour predictions)
   - Transformer models for sequence prediction

2. **Short-term Models** (1-30 days):
   - GRU networks with multi-task learning
   - Gradient Boosting (XGBoost, LightGBM with GPU)

3. **Long-term Models** (1-12 months):
   - Ensemble of LSTM + Fundamental analysis
   - Prophet for trend decomposition

4. **Pattern Detection**:
   - CNN for chart pattern recognition
   - Autoencoders for anomaly detection

---

### 3. Pattern Detection Service
**Purpose**: Identify technical chart patterns and market regimes

**Responsibilities**:
- Detect classic patterns (Head & Shoulders, Double Top/Bottom, Triangles, Flags)
- Identify support/resistance levels
- Candlestick pattern recognition
- Volume profile analysis
- Market regime classification (trending, ranging, volatile)

**Technology**:
- **Computer Vision**: ML.NET Image Classification
- **Signal Processing**: MathNet.Numerics for Fourier transforms, wavelets
- **Pattern Libraries**: Custom implementations + TA-Lib.NETCore
- **Cache**: Redis for detected patterns (TTL-based)

**Algorithms**:
- Sliding window convolution for pattern matching
- Dynamic Time Warping (DTW) for pattern similarity
- Hidden Markov Models for regime detection

---

### 4. Trend Detection Service
**Purpose**: Identify and classify market trends across timeframes

**Responsibilities**:
- Multi-timeframe trend analysis
- Trend strength calculation
- Trend reversal detection
- Moving average analysis (SMA, EMA, VWAP)
- Momentum indicators (RSI, MACD, Stochastic)
- Trend confidence scoring

**Technology**:
- **Technical Indicators**: TA-Lib.NETCore, Trady
- **Statistical Analysis**: MathNet.Numerics
- **Real-time Processing**: Reactive Extensions (Rx.NET)

**Analysis Layers**:
- **Primary Trend**: Weekly/Monthly charts (long-term)
- **Intermediate Trend**: Daily charts (short-term)
- **Minor Trend**: Hourly/Minute charts (intra-day)

---

### 5. Price Prediction Service
**Purpose**: Generate price predictions across multiple timeframes

**Responsibilities**:
- Aggregate predictions from multiple models
- Confidence interval calculation
- Risk assessment (volatility forecasting)
- Ensemble prediction with model weighting
- Prediction explanation (feature importance)

**Technology**:
- **Ensemble Methods**: Model averaging, stacking, blending
- **Uncertainty Quantification**: Monte Carlo dropout, quantile regression
- **Explainability**: SHAP values via ShapleyValues.NET

**Output**:
- Price targets (pessimistic, realistic, optimistic)
- Probability distributions
- Confidence scores
- Key drivers (most influential features)

---

### 6. Alert Processor Service
**Purpose**: Monitor conditions and generate alerts for trading signals

**Responsibilities**:
- Real-time condition monitoring
- Alert rule engine
- Notification delivery (email, SMS, push)
- Alert history and analytics
- Alert priority management

**Technology**:
- **Background Processing**: Hangfire for job scheduling
- **Rule Engine**: Custom DSL + Rules Engine (NRules)
- **Notifications**:
  - Email: MailKit
  - SMS: Twilio API
  - Push: SignalR for web notifications
- **Queue**: Redis Pub/Sub for alert distribution

**Alert Types**:
- Price targets hit
- Pattern detected
- Trend reversal
- Volatility spike
- Model prediction confidence threshold
- Custom user-defined rules

---

### 7. Backtesting Engine
**Purpose**: Validate strategies and models using historical data

**Responsibilities**:
- Historical simulation with walk-forward analysis
- Performance metrics calculation (Sharpe, Sortino, Max DD)
- Transaction cost modeling
- Slippage simulation
- Model performance over time
- Strategy optimization

**Technology**:
- **Backtesting Framework**: Custom implementation
- **Performance Analytics**: QuantConnect.Lean libraries (adapted)
- **Parallel Processing**: TPL Dataflow for parallel backtests
- **Results Storage**: SQL Server + MongoDB

**Metrics**:
- Win rate, profit factor
- Sharpe ratio, Sortino ratio
- Maximum drawdown
- Average trade duration
- Model accuracy over time windows

---

## Technology Stack

### Backend Stack (.NET 10)

#### Core Framework
- **ASP.NET Core 10**: Web API, SignalR for real-time
- **Entity Framework Core 10**: SQL Server ORM
- **Dapper**: High-performance queries
- **MediatR**: CQRS pattern implementation
- **AutoMapper**: Object-to-object mapping
- **FluentValidation**: Input validation

#### Background Processing
- **Quartz.NET**: Scheduled jobs (data fetching)
- **Hangfire**: Background jobs with UI dashboard
- **Polly**: Resilience and transient fault handling

#### ML & Analytics
- **ML.NET**: Machine learning framework
- **TorchSharp**: Deep learning with GPU support
- **ONNX Runtime**: Optimized model inference
- **MathNet.Numerics**: Statistical computing
- **TA-Lib.NETCore**: Technical analysis indicators
- **Trady**: Technical analysis library

#### Caching & Messaging
- **StackExchange.Redis**: Redis client
- **MassTransit**: Message bus (optional for future scaling)

---

### Frontend Stack

#### Primary UI: Blazor Server
**Why Blazor Server**:
- Real-time updates via SignalR (built-in)
- Full .NET stack integration
- Rich component ecosystem
- Low latency (running on same machine)

**Libraries**:
- **MudBlazor**: Material Design component library
- **Blazorise**: Additional UI components
- **Plotly.NET**: Interactive financial charts
- **TradingView Lightweight Charts**: Advanced charting

**Alternative**: Blazor WebAssembly for cloud hosting phase

---

### Data Storage

#### 1. Microsoft SQL Server (Primary Relational DB)
**Purpose**: Structured transactional data

**Schema Design**:
- **Stocks**: Symbol metadata, company info
- **OHLCV Data**: Historical bars (partitioned by date)
- **Predictions**: Model predictions with metadata
- **Alerts**: Alert configurations and history
- **Backtests**: Test results and metrics
- **Users**: User settings and preferences

**Features to Use**:
- **Columnstore Indexes**: For OHLCV data (10x compression)
- **Partitioning**: Partition by year/month for historical data
- **Temporal Tables**: Track data changes over time
- **In-Memory OLTP**: For hot cache tables

**Edition**: SQL Server 2022 Developer (free) or Standard

---

#### 2. MongoDB (NoSQL Document Store)
**Purpose**: Semi-structured and high-volume data

**Collections**:
- **RawMarketData**: Raw API responses (audit trail)
- **NewsArticles**: Financial news with sentiment
- **ModelArtifacts**: Trained model metadata
- **TickData**: Intra-day tick-level data (optional)
- **PatternInstances**: Detected patterns with images
- **FeatureStore**: Pre-computed features for ML

**Features to Use**:
- **Time Series Collections**: Optimized for time-series data (MongoDB 5.0+)
- **Aggregation Pipeline**: For complex analytics
- **Change Streams**: Real-time data monitoring
- **GridFS**: Store large model files and chart images

**Sizing**: Start with single instance, shard by symbol if needed

---

#### 3. Redis (In-Memory Cache & Messaging)
**Purpose**: High-speed caching and real-time data

**Use Cases**:
- **Cache Layer**:
  - Recent stock quotes (1-5 min TTL)
  - Computed indicators (15-30 min TTL)
  - User session data
  - Model predictions (1-hour TTL)

- **Pub/Sub**:
  - Real-time quote distribution
  - Alert notifications
  - Model update events

- **Data Structures**:
  - Sorted Sets: Leaderboards (top gainers/losers)
  - Lists: Recent alerts queue
  - Hashes: Stock metadata

**Configuration**:
- Redis 7.x with persistence (RDB + AOF)
- 8-16 GB allocation

---

### Infrastructure Components

#### Message Queue (Future)
- **RabbitMQ** or **Azure Service Bus**: When scaling to cloud

#### Logging & Monitoring
- **Serilog**: Structured logging to files, SQL, Seq
- **Seq**: Log aggregation and search (self-hosted)
- **Prometheus + Grafana**: Metrics and dashboards
- **Application Insights**: When using Azure

#### API Gateway (Future Cloud Phase)
- **YARP (Yet Another Reverse Proxy)**: .NET-based reverse proxy
- **Azure API Management**: Cloud phase

---

## Analysis Timeframes

### 1. Intra-Day Analysis
**Timeframe**: Minutes to hours (same trading day)

**Data Granularity**: 1-min, 5-min, 15-min bars

**Models**:
- LSTM with 60-min lookback
- Transformer with attention (30-min predictions)
- XGBoost for 5-15 min ahead

**Features**:
- Price action (OHLCV)
- Volume profile, VWAP
- Order flow (if available)
- Momentum oscillators (fast RSI, stochastic)
- Market breadth indicators

**Update Frequency**: Real-time (1-5 min)

**Use Cases**:
- Day trading signals
- Scalping opportunities
- Intra-day volatility alerts

---

### 2. Short-Term Analysis
**Timeframe**: 1 day to 30 days

**Data Granularity**: 1-hour, 4-hour, daily bars

**Models**:
- GRU with multi-task learning (3, 5, 10-day predictions)
- Ensemble: LightGBM + LSTM
- Pattern recognition models

**Features**:
- Technical indicators (20-day MA, RSI, MACD)
- Volume trends
- Support/resistance levels
- Sector rotation indicators
- Earnings calendar proximity

**Update Frequency**: Hourly to daily

**Use Cases**:
- Swing trading
- Weekly options strategies
- Short-term trend following

---

### 3. Long-Term Analysis
**Timeframe**: 1 month to 12 months

**Data Granularity**: Daily, weekly bars

**Models**:
- Prophet for trend + seasonality
- LSTM with fundamental features
- Random Forest with macro indicators

**Features**:
- Fundamental ratios (P/E, P/B, ROE)
- Macro indicators (GDP, inflation, rates)
- Sector performance
- Long-term moving averages (50-day, 200-day)
- Seasonal patterns

**Update Frequency**: Daily to weekly

**Use Cases**:
- Investment portfolio allocation
- Long-term trend identification
- Risk management

---

## Data Sources & Costs

### Recommended Paid Data Providers

#### Primary: Polygon.io
**Why**: Best value for real-time + historical data

**Plan**: Stocks Starter ($99/month) or Advanced ($199/month)
- Real-time stock quotes (unlimited)
- Historical data (all US stocks)
- 1-min aggregates
- WebSocket streaming
- News and sentiment data
- API rate: 5 calls/sec (Starter), 100 calls/sec (Advanced)

**Coverage**: All US stocks, 50+ years historical

---

#### Alternative Primary: Alpha Vantage
**Plan**: Premium ($49.99/month) or Ultimate ($149.99/month)
- Real-time and historical data
- Technical indicators (pre-computed)
- Fundamental data
- News sentiment
- API rate: 75 calls/min (Premium), 1200 calls/min (Ultimate)

**Limitation**: Historical data limited to 20 years

---

#### Backup/Free: Yahoo Finance (via YFinance.NET)
**Cost**: Free
**Limitations**:
- No official API (web scraping risk)
- Rate limiting
- No real-time guarantees
- Good for backtesting historical data

---

#### News & Sentiment: NewsAPI.org
**Plan**: Business ($449/month)
- Global news coverage
- Sentiment analysis
- Historical news archive
- 250,000 requests/month

---

#### Alternative News: Benzinga
**Plan**: News API ($500/month)
- Financial news feed
- Earnings calendars
- Analyst ratings
- Real-time press releases

---

#### Fundamental Data: Financial Modeling Prep
**Plan**: Professional ($14/month) or Enterprise ($69/month)
- Financial statements
- Key metrics and ratios
- Insider trades
- Stock screener data

---

### Recommended Provider Strategy

**Phase 1 (Initial Development)**:
- **Primary**: Polygon.io Stocks Starter ($99/month)
- **Fundamental**: Financial Modeling Prep Professional ($14/month)
- **Backup**: Yahoo Finance (free)
- **Total**: ~$113/month

**Phase 2 (Production)**:
- **Primary**: Polygon.io Stocks Advanced ($199/month)
- **News**: NewsAPI.org Business ($449/month) or Benzinga ($500/month)
- **Fundamental**: Financial Modeling Prep Enterprise ($69/month)
- **Total**: ~$717-768/month

---

## Backtesting Strategy

### Backtesting Framework Architecture

```
Historical Data → Feature Engineering → Model Training (Walk-Forward)
                                              ↓
                                    Generate Signals
                                              ↓
                                    Simulate Trades
                                              ↓
                        Calculate Performance Metrics
                                              ↓
                                    Optimization Loop
```

### Implementation Approach

#### 1. Walk-Forward Analysis
**Purpose**: Avoid look-ahead bias and overfitting

**Method**:
```
Training Window: 252 days (1 year)
Validation Window: 63 days (3 months)
Test Window: 21 days (1 month)

Timeline:
[---Train (252)---][--Val (63)--][Test (21)]
    ↓
    [---Train (252)---][--Val (63)--][Test (21)]
        ↓
        [---Train (252)---][--Val (63)--][Test (21)]
```

**Process**:
1. Train model on training window
2. Validate hyperparameters on validation window
3. Test on unseen test window
4. Roll window forward
5. Aggregate results across all test windows

---

#### 2. Simulation Components

**Transaction Costs**:
- Commission: $0 (most brokers) or $0.005/share
- Slippage: 0.05% for liquid stocks, 0.1-0.5% for illiquid
- Spread: Bid-ask spread from historical data or estimated

**Order Execution**:
- Market orders: Fill at next bar's open
- Limit orders: Fill if price touches limit within bar
- Stop orders: Fill if triggered within bar

**Position Sizing**:
- Fixed dollar amount
- Fixed percentage of portfolio
- Kelly Criterion
- Volatility-based (ATR)

---

#### 3. Performance Metrics

**Returns**:
- Total return
- Annualized return
- Monthly/quarterly returns
- Benchmark comparison (S&P 500)

**Risk Metrics**:
- **Sharpe Ratio**: (Return - RiskFreeRate) / StdDev
- **Sortino Ratio**: (Return - RiskFreeRate) / DownsideDeviation
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: AnnualReturn / MaxDrawdown
- **Value at Risk (VaR)**: 95th percentile loss
- **Conditional VaR**: Expected loss beyond VaR

**Trade Metrics**:
- Win rate (% profitable trades)
- Average win / Average loss
- Profit factor (Gross profit / Gross loss)
- Average trade duration
- Max consecutive losses

**Model Metrics**:
- Prediction accuracy (directional)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

---

#### 4. Backtesting Data Requirements

**Minimum Historical Data**:
- **Intra-day models**: 6 months of minute data
- **Short-term models**: 3-5 years of daily data
- **Long-term models**: 10+ years of daily data

**Data Quality Checks**:
- Handle stock splits and dividends (adjust prices)
- Filter for survivorship bias (include delisted stocks)
- Validate data integrity (check for gaps, outliers)
- Corporate actions handling

---

#### 5. Backtesting Tools & Libraries

**Custom Framework** (Recommended):
- Full control over simulation logic
- Optimized for GPU-accelerated models
- Integration with existing data pipeline

**Components**:
```csharp
// Pseudo-structure
IBacktestEngine
├── IDataProvider (historical data)
├── ISignalGenerator (model predictions)
├── IPortfolioManager (position sizing, risk)
├── IOrderExecutor (simulate fills)
├── IPerformanceCalculator (metrics)
└── IResultsReporter (visualization)
```

**Third-party Libraries** (Reference):
- **QuantConnect.Lean**: Open-source algorithmic trading engine
  - Can extract backtesting components
  - Has .NET implementation

---

#### 6. Optimization Strategy

**Hyperparameter Optimization**:
- **Grid Search**: Exhaustive search over parameter space
- **Random Search**: Sample random combinations
- **Bayesian Optimization**: Optuna.NET (intelligent search)
- **Genetic Algorithms**: Evolve best parameters

**Cross-Validation**:
- Time-series cross-validation (walk-forward)
- Purged cross-validation (remove temporal leakage)

**Overfitting Prevention**:
- Separate validation and test sets
- Regularization in models
- Limit number of parameters
- Ensemble multiple models

---

## Deployment Architecture

### Phase 1: Local Workstation (Current Phase)

```
┌─────────────────────────────────────────────────────────┐
│              Local Workstation (Windows 11)             │
│                                                         │
│  ┌───────────────────────────────────────────────┐     │
│  │        .NET 10 Application (Self-Hosted)      │     │
│  │  ┌─────────┐  ┌──────────┐  ┌─────────────┐  │     │
│  │  │ Blazor  │  │ REST API │  │ Background  │  │     │
│  │  │ Server  │  │ Services │  │ Services    │  │     │
│  │  │ (5000)  │  │ (5001)   │  │ (Quartz)    │  │     │
│  │  └─────────┘  └──────────┘  └─────────────┘  │     │
│  │                                               │     │
│  │  ┌─────────────────────────────────────┐     │     │
│  │  │   ML Pipeline (GPU-Accelerated)     │     │     │
│  │  │   - TorchSharp + CUDA 12.x          │     │     │
│  │  │   - ONNX Runtime GPU                │     │     │
│  │  └─────────────────────────────────────┘     │     │
│  └───────────────────────────────────────────────┘     │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌─────────────────┐      │
│  │ SQL      │  │ MongoDB  │  │ Redis           │      │
│  │ Server   │  │ 6.0+     │  │ 7.x             │      │
│  │ 2022     │  │          │  │                 │      │
│  └──────────┘  └──────────┘  └─────────────────┘      │
│                                                         │
│  Hardware: NVIDIA RTX 3090, 64GB RAM, NVMe SSD         │
└─────────────────────────────────────────────────────────┘
                         │
                         ↓
              External Data APIs
         (Polygon.io, Alpha Vantage)
```

**Ports**:
- Blazor UI: 5000 (HTTP), 5001 (HTTPS)
- Hangfire Dashboard: 5002
- Seq Logging: 5341
- SQL Server: 1433
- MongoDB: 27017
- Redis: 6379

**Security**:
- Windows Firewall: Block external access
- HTTPS with self-signed cert (local)
- API keys in User Secrets / Environment Variables

---

### Phase 2: Hybrid Cloud (Future)

```
┌──────────────────────────────────────────────────────┐
│                  Azure Cloud                         │
│                                                      │
│  ┌────────────────────────────────────────────┐     │
│  │         App Service (Blazor UI)            │     │
│  │         - Auto-scale                       │     │
│  │         - Custom domain                    │     │
│  └────────────────────────────────────────────┘     │
│                       │                              │
│  ┌────────────────────────────────────────────┐     │
│  │      Azure API Management                  │     │
│  │      - Rate limiting                       │     │
│  │      - OAuth 2.0                           │     │
│  └────────────────────────────────────────────┘     │
│                                                      │
│  ┌────────────────────────────────────────────┐     │
│  │      Azure Container Apps                  │     │
│  │      - Background services                 │     │
│  │      - Auto-scale                          │     │
│  └────────────────────────────────────────────┘     │
│                                                      │
│  ┌────────────┐  ┌──────────────────────────┐      │
│  │ Azure SQL  │  │ Cosmos DB (MongoDB API)  │      │
│  │ Database   │  │                          │      │
│  └────────────┘  └──────────────────────────┘      │
│                                                      │
│  ┌────────────┐  ┌──────────────────────────┐      │
│  │ Azure      │  │ Application Insights     │      │
│  │ Cache      │  │ (Monitoring)             │      │
│  │ (Redis)    │  │                          │      │
│  └────────────┘  └──────────────────────────┘      │
└──────────────────────────────────────────────────────┘
                      ↕
┌──────────────────────────────────────────────────────┐
│            Local Workstation                         │
│  ┌────────────────────────────────────────────┐     │
│  │   ML Training & Inference (GPU)            │     │
│  │   - Model training (batch jobs)            │     │
│  │   - Inference endpoint (local)             │     │
│  │   - Pushes models to Azure Blob            │     │
│  └────────────────────────────────────────────┘     │
│                                                      │
│  Hardware: NVIDIA RTX 3090                           │
└──────────────────────────────────────────────────────┘
```

**Rationale for Hybrid**:
- GPU training/inference on local workstation (cost-effective)
- UI and APIs on Azure (availability, scalability)
- Data stored in cloud (backup, accessibility)
- ML models synced to Azure Blob Storage

---

## Azure Cloud Services & Cost Estimates

### Recommended Azure Services

#### 1. Compute

**Azure App Service (UI Hosting)**
- **SKU**: S1 Standard
  - 1 core, 1.75 GB RAM
  - Custom domain, SSL
  - Auto-scale up to 10 instances
- **Cost**: ~$73/month (S1)

**Azure Container Apps (Background Services)**
- **Configuration**: 2 apps (Data Fetcher, Alert Processor)
  - 0.5 vCPU, 1 GB RAM each
  - Consumption-based pricing
- **Cost**: ~$30-50/month (estimated)

**Alternative: Azure Kubernetes Service (AKS)** - Overkill for single user

---

#### 2. Data Storage

**Azure SQL Database**
- **SKU**: General Purpose (Serverless)
  - 2 vCores, auto-pause
  - 32 GB storage
- **Cost**: ~$85/month (serverless, avg usage)

**Azure Cosmos DB (MongoDB API)**
- **SKU**: Provisioned throughput
  - 400 RU/s (autoscale to 4000 RU/s)
  - 50 GB storage
- **Cost**: ~$25/month (400 RU/s) + $0.25/GB = ~$38/month

**Alternative: MongoDB Atlas** (Azure Marketplace)
- M10 tier: $57/month (10 GB storage)

**Azure Cache for Redis**
- **SKU**: Basic C1 (1 GB)
- **Cost**: ~$18/month

**Azure Blob Storage (Model artifacts, backups)**
- **Tier**: Hot
- **Storage**: 100 GB
- **Cost**: ~$2/month

---

#### 3. Monitoring & Management

**Application Insights**
- **Ingestion**: ~5 GB/month
- **Cost**: ~$12/month (first 5 GB free, then $2.30/GB)

**Azure Monitor**
- **Metrics**: Included
- **Logs**: ~5 GB/month
- **Cost**: ~$3/month

**Azure Key Vault**
- **Operations**: Standard tier
- **Cost**: ~$1/month

---

#### 4. Networking

**Azure API Management**
- **SKU**: Consumption tier
  - Pay-per-execution
  - Up to 1M calls
- **Cost**: ~$4/month (low traffic)

**Azure Virtual Network**
- **Cost**: Free (for standard VNet)

---

### Total Azure Cost Estimate (Hybrid Phase)

| Service | Monthly Cost |
|---------|--------------|
| App Service (S1) | $73 |
| Container Apps | $40 |
| Azure SQL (Serverless) | $85 |
| Cosmos DB (MongoDB) | $38 |
| Azure Cache (Redis C1) | $18 |
| Blob Storage | $2 |
| Application Insights | $12 |
| Azure Monitor | $3 |
| Key Vault | $1 |
| API Management (Consumption) | $4 |
| **Total** | **~$276/month** |

**Plus Data Costs**:
- Polygon.io: $199/month
- Financial Modeling Prep: $69/month
- NewsAPI: $449/month (optional)

**Grand Total**: ~$545-994/month (without news) or (with news)

---

### Cost Optimization Strategies

**Phase 1 (Local Only)**:
- **Zero Azure costs**
- Data providers: $113/month (Polygon Starter + FMP)
- **Total**: $113/month

**Phase 2a (Hybrid, Dev/Test)**:
- Use Azure Dev/Test pricing (free credits if MSDN subscriber)
- Serverless SQL pauses when not used
- Container Apps scale to zero
- **Estimated**: $150-200/month Azure + $113 data = $263-313/month

**Phase 2b (Hybrid, Production)**:
- Full deployment as above
- **Estimated**: $276 Azure + $717 data = $993/month

**Future Optimization**:
- Reserved instances (1-year commit): 30-40% discount
- Spot instances for batch ML training
- Blob storage in Cool tier for old data

---

## Workstation Specifications

### Hardware Requirements for ML Workstation

#### Best Configuration (Optimal Performance)
**Target**: Maximum ML training speed, large models, minimal wait times

| Component | Specification | Price (USD) |
|-----------|---------------|-------------|
| **GPU** | NVIDIA RTX 4090 (24 GB VRAM) | $1,599 |
| **CPU** | AMD Ryzen 9 7950X (16-core) or Intel i9-13900K | $550 |
| **RAM** | 128 GB DDR5-5600 (4x32GB) | $400 |
| **Storage** | 2 TB NVMe Gen4 SSD (OS + Apps) | $150 |
|           | 4 TB NVMe Gen4 SSD (Data + Models) | $280 |
| **Motherboard** | X670E or Z790 chipset | $350 |
| **PSU** | 1000W 80+ Gold | $180 |
| **Cooling** | AIO 360mm liquid cooler | $150 |
| **Case** | Full tower with airflow | $150 |
| **Total** | | **~$3,809** |

**Performance**:
- Train LSTM models in minutes instead of hours
- Support for multi-GPU in future (RTX 4090 x2)
- Handle large transformer models (up to 20GB)

---

#### Recommended Configuration (Best Value)
**Target**: Your current setup - excellent for this use case

| Component | Specification | Price (USD) |
|-----------|---------------|-------------|
| **GPU** | NVIDIA RTX 3090 (24 GB VRAM) ✓ (You have) | $1,099 (used) |
| **CPU** | AMD Ryzen 9 5950X (16-core) or Intel i7-12700K | $400 |
| **RAM** | 64 GB DDR4-3600 (4x16GB or 2x32GB) | $180 |
| **Storage** | 1 TB NVMe Gen3 SSD (OS + Apps) | $80 |
|           | 2 TB NVMe Gen3 SSD (Data + Models) | $130 |
| **Motherboard** | X570 or Z690 chipset | $230 |
| **PSU** | 850W 80+ Gold | $130 |
| **Cooling** | AIO 280mm or good air cooler | $90 |
| **Case** | Mid-tower with airflow | $90 |
| **Total** | | **~$2,429** |

**Performance**:
- RTX 3090 is perfect for this workload (24GB VRAM)
- 64 GB RAM sufficient for most models and backtesting
- Fast storage for large datasets

**Your Current Setup Analysis**:
If you have RTX 3090, you're already well-equipped. Ensure:
- At least 64 GB RAM (32 GB minimum)
- Fast NVMe SSD (1 TB+ for OS, 2 TB+ for data)
- Good CPU (8+ cores, 3.5+ GHz boost)
- 850W+ PSU to handle 3090's 350W TDP

---

#### Medium Configuration (Budget-Friendly)
**Target**: Entry-level ML, slower training, cost-conscious

| Component | Specification | Price (USD) |
|-----------|---------------|-------------|
| **GPU** | NVIDIA RTX 3060 Ti (8 GB VRAM) or RTX 4060 Ti 16GB | $450-550 |
| **CPU** | AMD Ryzen 7 5700X (8-core) or Intel i5-12600K | $250 |
| **RAM** | 32 GB DDR4-3200 (2x16GB) | $80 |
| **Storage** | 500 GB NVMe SSD (OS + Apps) | $45 |
|           | 1 TB NVMe SSD (Data + Models) | $80 |
| **Motherboard** | B550 or B660 chipset | $150 |
| **PSU** | 650W 80+ Bronze | $70 |
| **Cooling** | Air cooler (Hyper 212 or equivalent) | $40 |
| **Case** | Mid-tower budget case | $60 |
| **Total** | | **~$1,225-1,325** |

**Performance**:
- Can train smaller models (LSTM, GRU)
- 8 GB VRAM limits large transformers
- Longer training times (3-5x slower than 3090)
- Good for learning and prototyping

**Limitations**:
- May struggle with large ensemble models
- Cannot train very deep networks or large transformers
- Recommend RTX 4060 Ti 16GB ($550) if budget allows

---

### GPU Comparison for ML Workloads

| GPU Model | VRAM | CUDA Cores | FP32 TFLOPS | Price | Best For |
|-----------|------|------------|-------------|-------|----------|
| RTX 4090 | 24 GB | 16,384 | 82.6 | $1,599 | Large models, fastest training |
| RTX 3090 | 24 GB | 10,496 | 35.6 | $1,099 | **Your current - Perfect for this** |
| RTX 4080 | 16 GB | 9,728 | 48.7 | $1,199 | Fast but limited VRAM |
| RTX 3080 Ti | 12 GB | 10,240 | 34.1 | $750 | Good value, moderate VRAM |
| RTX 4060 Ti 16GB | 16 GB | 4,352 | 22.1 | $550 | Budget option with good VRAM |
| RTX 3060 Ti | 8 GB | 4,864 | 16.2 | $450 | Entry-level ML |

**Key Considerations**:
- **VRAM**: Most important for ML (24 GB ideal, 16 GB good, 8 GB minimum)
- **CUDA Cores**: Higher = faster training
- **Tensor Cores**: All RTX have them (crucial for deep learning)

**Your RTX 3090 is Excellent Because**:
- 24 GB VRAM: Can handle large models and batch sizes
- Good performance (35 TFLOPS FP32)
- Mature CUDA ecosystem
- Only RTX 4090 is significantly better (2.3x faster, but 45% more expensive)

---

### Storage Considerations

**NVMe SSD Recommendations**:

**OS & Applications** (500 GB - 1 TB):
- Samsung 980 Pro 1TB (~$80)
- WD Black SN850X 1TB (~$90)
- Kingston KC3000 1TB (~$75)

**Data & Models** (2 TB - 4 TB):
- Samsung 990 Pro 2TB (~$150)
- WD Black SN850X 2TB (~$130)
- Crucial T700 2TB (~$140)

**Why NVMe is Critical**:
- Loading large datasets: 10x faster than SATA SSD
- Model checkpointing: Fast saves during training
- Backtesting: Rapid sequential reads of historical data

**Storage Layout**:
```
Drive 1 (1 TB NVMe): OS + Applications + SQL Server
Drive 2 (2-4 TB NVMe):
  - Historical market data (500 GB - 2 TB)
  - Trained models (50-100 GB)
  - MongoDB data (100-500 GB)
  - Backtesting results (50 GB)
```

---

### RAM Requirements

**32 GB (Minimum)**:
- Can run medium-sized models
- Limited concurrent backtesting
- May need to batch large datasets

**64 GB (Recommended)**:
- Comfortable for most workloads
- Load full datasets in memory
- Run multiple experiments simultaneously
- Good for your RTX 3090

**128 GB (Optimal)**:
- Load years of minute-level data in RAM
- Train very large models
- Parallel backtesting across symbols
- Overkill unless processing 1000+ stocks

**Rule of Thumb**:
- Match your GPU VRAM: RTX 3090 (24 GB) → 64 GB RAM
- Data-intensive tasks: 2-3x your dataset size

---

### Power & Cooling

**Power Supply**:
- RTX 3090 TDP: 350W
- CPU (16-core): 150-200W
- System overhead: 100W
- **Minimum**: 750W
- **Recommended**: 850W (headroom for spikes)
- **Optimal**: 1000W (future-proof)

**Cooling**:
- RTX 3090 runs hot (85°C+ under load)
- Good case airflow essential
- CPU cooler: AIO 280mm+ or high-end air cooler
- Consider case fans: 3x intake, 2x exhaust

---

### Network

**Internet Connection**:
- **Download**: 100+ Mbps (for data fetching)
- **Upload**: 10+ Mbps (for cloud sync)
- **Latency**: <50ms (for real-time quotes)

**Uninterruptible Power Supply (UPS)**:
- Recommended: 1500VA / 900W
- Protects during training runs
- Cost: $150-200

---

## Future Phases

### Phase 3: Cryptocurrency Integration

**Additional Components**:
- **Crypto Exchanges API**: Binance, Coinbase Pro, Kraken
- **On-Chain Data**: Glassnode, CryptoQuant
- **24/7 Operation**: Always-on market (no trading hours)

**Challenges**:
- Higher volatility requires different models
- 24/7 monitoring and alerting
- Faster price movements (1-second bars)

**Cost**:
- Exchange API: Free to $50/month
- On-chain data: $300-800/month (Glassnode)

---

### Phase 4: Multi-User / SaaS

**Architecture Changes**:
- Multi-tenancy
- User authentication (Azure AD B2C)
- Subscription billing (Stripe)
- Rate limiting per user
- Shared models, isolated predictions

**Azure Services**:
- Azure Front Door (CDN, WAF)
- Azure SQL Elastic Pool
- Cosmos DB autoscale
- Azure Functions (event-driven)

**Estimated Cost** (100 users):
- Azure: $800-1200/month
- Data: Negotiate bulk pricing

---

## Next Steps & Priorities

### Phase 1 Milestones

1. **Setup Development Environment** (Week 1)
   - Install .NET 10, SQL Server, MongoDB, Redis
   - Configure NVIDIA drivers, CUDA 12.x, cuDNN
   - Setup Polygon.io API account
   - Initialize Git repository

2. **Data Infrastructure** (Week 2-3)
   - Design database schemas (SQL Server, MongoDB)
   - Implement Data Fetcher Service
   - Build data validation pipeline
   - Setup Redis caching layer

3. **ML Pipeline MVP** (Week 4-6)
   - Implement feature engineering
   - Train baseline LSTM model (short-term)
   - Setup ONNX Runtime for inference
   - Build model versioning system

4. **Core Services** (Week 7-9)
   - Implement Trend Detection Service
   - Implement Pattern Detection Service
   - Implement Price Prediction Service
   - Build alert rule engine

5. **Backtesting Framework** (Week 10-11)
   - Build historical simulation engine
   - Implement performance metrics
   - Create walk-forward validation
   - Generate backtest reports

6. **UI Development** (Week 12-14)
   - Setup Blazor Server project
   - Build dashboard with charts
   - Implement real-time updates
   - Create alert management UI

7. **Testing & Optimization** (Week 15-16)
   - Performance testing
   - Model accuracy validation
   - UI/UX refinement
   - Documentation

---

## Risk Mitigation

### Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| API rate limits | Data gaps | Multi-provider failover, caching |
| Model overfitting | Poor predictions | Walk-forward validation, regularization |
| GPU memory errors | Training failures | Batch size tuning, gradient checkpointing |
| Data quality issues | Bad predictions | Validation pipeline, outlier detection |
| System downtime | Missed opportunities | UPS, auto-restart, monitoring |

### Operational Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| High data costs | Budget overrun | Start with cheaper tier, optimize fetching |
| Slow model training | Delayed iterations | Use smaller models initially, optimize code |
| Storage capacity | System failure | Monitor disk usage, archive old data |
| Internet outage | No data updates | Fallback to cached data, offline mode |

---

## Conclusion

This architecture provides a solid foundation for a GPU-accelerated stock market prediction application running on your local workstation. The design leverages your RTX 3090 for ML workloads while keeping costs low in Phase 1. The hybrid cloud strategy in Phase 2 allows you to scale the UI and data storage while keeping compute-intensive ML tasks local.

**Key Strengths**:
- Modern .NET 10 stack with excellent GPU support
- Scalable data architecture (SQL + NoSQL + Redis)
- Robust backtesting framework
- Clear path to cloud migration
- Cost-effective development approach

**Recommended Next Steps**:
1. Review and approve architecture
2. Setup development environment
3. Start with Data Fetcher Service and database schemas
4. Build ML pipeline MVP with one model
5. Iterate and add features incrementally

---

**Document Version**: 1.0
**Last Updated**: 2026-02-03
**Author**: Architecture Planning Phase
