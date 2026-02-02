# Stock Market Price Prediction Application - Technical Specification

**Version:** 1.0
**Date:** 2026-02-02
**Status:** Draft

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Architecture](#3-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Data Layer](#5-data-layer)
6. [API Specification](#6-api-specification)
7. [Frontend Specification](#7-frontend-specification)
8. [ML/AI Pipeline](#8-mlai-pipeline)
9. [Background Jobs](#9-background-jobs)
10. [Security](#10-security)
11. [Infrastructure](#11-infrastructure)
12. [Cost Estimates](#12-cost-estimates)
13. [Development Phases](#13-development-phases)
14. [Future Expansion](#14-future-expansion)

---

## 1. Executive Summary

### 1.1 Project Overview

A cloud-native stock market price prediction application that enables users to:
- **Download** historical and current stock prices
- **Predict** future price movements across multiple timeframes
- **Identify** technical patterns and trading signals

### 1.2 Key Decisions

| Aspect | Decision |
|--------|----------|
| Target Market | US markets (NYSE, NASDAQ, AMEX) |
| Data Frequency | End-of-day (EOD) |
| Prediction Horizons | Short (1-5 days), Medium (1-4 weeks), Long (1-6 months) |
| User Base | Retail investors (educational focus) |
| Monetization | Freemium model |
| Cloud Platform | Microsoft Azure |

### 1.3 Success Criteria

- Support 8,000+ US stock symbols
- Process daily updates within 30 minutes of market close
- Prediction accuracy > baseline (buy-and-hold) by 5%+
- System uptime > 99.5%
- Response time < 500ms for API calls

---

## 2. System Overview

### 2.1 Core Features

```
┌─────────────────────────────────────────────────────────────────┐
│                      CORE FEATURES                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PRICE DATA                                                     │
│  ├── Real-time price display (delayed 15-20 min for free)      │
│  ├── Historical OHLCV data (Open, High, Low, Close, Volume)    │
│  ├── Adjustments for splits and dividends                      │
│  └── Data from multiple providers for redundancy               │
│                                                                 │
│  PREDICTIONS                                                    │
│  ├── Short-term (1-5 day) price direction                      │
│  ├── Medium-term (1-4 week) trend analysis                     │
│  ├── Long-term (1-6 month) investment outlook                  │
│  ├── Confidence scores for each prediction                     │
│  └── Historical prediction accuracy tracking                   │
│                                                                 │
│  PATTERN DETECTION                                              │
│  ├── Candlestick patterns (Doji, Hammer, Engulfing, etc.)      │
│  ├── Chart patterns (Head & Shoulders, Triangles, etc.)        │
│  ├── Technical indicator signals (RSI, MACD, Moving Averages)  │
│  └── Support/Resistance level detection                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 User Tiers (Retail Focus)

#### Free Tier
- Watchlist: Up to 10 stocks
- Predictions: Basic (bullish/bearish/neutral) for short-term only
- Charts: Simple line/candlestick with volume
- Patterns: Top 3 detected patterns
- Alerts: None
- Data: 1 year history
- Educational: Basic tooltips and explanations

#### Premium Tier ($9.99/month)
- Watchlist: Unlimited stocks
- Predictions: All timeframes with confidence scores
- Charts: Full technical indicators (RSI, MACD, Moving Averages)
- Patterns: All patterns with educational explanations
- Alerts: Email notifications, up to 20 active
- Data: 5 years history
- Educational: Full learning center access
- Support: Priority email support

---

## 3. Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              CLIENTS                                     │
├─────────────────────────────────────────────────────────────────────────┤
│  React SPA          │  Future: Mobile App    │  Future: API Clients     │
│  (Static Web Apps)  │  (React Native)        │  (Pro tier)              │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │ HTTPS
                          ┌────────▼────────┐
                          │  Azure Front    │
                          │  Door (CDN+WAF) │
                          └────────┬────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────┐
│                            API LAYER                                     │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐             │
│  │  .NET 10 API   │  │  Azure AD B2C  │  │  API Mgmt      │             │
│  │  (App Service) │  │  (Auth)        │  │  (Rate Limit)  │             │
│  └───────┬────────┘  └────────────────┘  └────────────────┘             │
│          │                                                               │
│          ▼                                                               │
│  ┌────────────────────────────────────────────────────────────┐         │
│  │  Application Insights (Monitoring, Logging, Tracing)       │         │
│  └────────────────────────────────────────────────────────────┘         │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────┐
│                          SERVICE LAYER                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐             │
│  │  Price         │  │  Prediction    │  │  Pattern       │             │
│  │  Service       │  │  Service       │  │  Service       │             │
│  └────────────────┘  └────────────────┘  └────────────────┘             │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐             │
│  │  User          │  │  Watchlist     │  │  Alert         │             │
│  │  Service       │  │  Service       │  │  Service       │             │
│  └────────────────┘  └────────────────┘  └────────────────┘             │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────┐
│                       BACKGROUND PROCESSING                              │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐             │
│  │  Data Fetcher  │  │  ML Inference  │  │  Pattern       │             │
│  │  (Functions)   │  │  (Container)   │  │  Detection     │             │
│  └────────────────┘  └────────────────┘  └────────────────┘             │
│                             │                                            │
│  ┌────────────────┐         │         ┌────────────────┐                │
│  │  ML Training   │◀────────┘         │  Alert         │                │
│  │  (Container)   │                   │  Processor     │                │
│  └────────────────┘                   └────────────────┘                │
│                                                                          │
│  Orchestration: Azure Durable Functions                                  │
│  Messaging: Azure Service Bus                                            │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────┐
│                           DATA LAYER                                     │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐             │
│  │  Azure SQL     │  │  Cosmos DB     │  │  Blob Storage  │             │
│  │  Serverless    │  │  (NoSQL)       │  │                │             │
│  │                │  │                │  │                │             │
│  │  - Users       │  │  - Prices      │  │  - ML Models   │             │
│  │  - Watchlists  │  │  - Predictions │  │  - Raw Data    │             │
│  │  - Alerts      │  │  - Patterns    │  │  - Backups     │             │
│  │  - Billing     │  │  - Indicators  │  │  - Logs        │             │
│  └────────────────┘  └────────────────┘  └────────────────┘             │
│                                                                          │
│  ┌────────────────┐                                                      │
│  │  Redis Cache   │  (Session state, hot data, rate limiting)           │
│  └────────────────┘                                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      DAILY DATA PIPELINE                                 │
│                    (Triggered at 4:30 PM ET)                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Step 1: FETCH                                                           │
│  ├── Trigger: Timer (Azure Function)                                     │
│  ├── Source: Alpha Vantage / Polygon.io API                              │
│  ├── Symbols: ~8,000 US stocks                                           │
│  ├── Batch Size: 100 symbols per request                                 │
│  └── Output: Raw JSON → Service Bus                                      │
│                                                                          │
│  Step 2: STORE                                                           │
│  ├── Trigger: Service Bus message                                        │
│  ├── Process: Validate, normalize, transform                             │
│  ├── Destination: Cosmos DB (prices container)                           │
│  └── Partition: By symbol (e.g., /symbol = "AAPL")                       │
│                                                                          │
│  Step 3: CALCULATE                                                       │
│  ├── Trigger: Storage complete event                                     │
│  ├── Calculate: Technical indicators (RSI, MACD, SMA, EMA, BB)           │
│  ├── Destination: Cosmos DB (indicators container)                       │
│  └── Cache: Redis for hot data                                           │
│                                                                          │
│  Step 4: PREDICT                                                         │
│  ├── Trigger: Indicators ready                                           │
│  ├── Models: Short-term, Medium-term, Long-term                          │
│  ├── Runtime: ML.NET inference in API / Python containers                │
│  ├── Output: Predictions with confidence scores                          │
│  └── Destination: Cosmos DB (predictions container)                      │
│                                                                          │
│  Step 5: DETECT                                                          │
│  ├── Trigger: Predictions complete                                       │
│  ├── Scan: Candlestick patterns, chart patterns                          │
│  ├── Output: Detected patterns with metadata                             │
│  └── Destination: Cosmos DB (patterns container)                         │
│                                                                          │
│  Step 6: ALERT                                                           │
│  ├── Trigger: Patterns detected                                          │
│  ├── Match: User alert rules against signals                             │
│  ├── Channels: Email (SendGrid), Push (Azure Notification Hub)           │
│  └── Log: Alert history in Azure SQL                                     │
│                                                                          │
│  Total Pipeline Duration: ~20-30 minutes                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Technology Stack

### 4.1 Frontend

| Component | Technology | Version | Notes |
|-----------|------------|---------|-------|
| Framework | React | 18.x | With TypeScript |
| Build Tool | Vite | 5.x | Fast builds |
| State Management | Zustand | 4.x | Lightweight |
| UI Components | Shadcn/ui | Latest | Tailwind-based |
| Charts | TradingView Lightweight Charts | 4.x | Financial charts |
| HTTP Client | TanStack Query | 5.x | Caching, retries |
| Forms | React Hook Form | 7.x | + Zod validation |
| Routing | React Router | 6.x | SPA routing |
| Testing | Vitest + Testing Library | Latest | Unit/integration |

### 4.2 Backend

| Component | Technology | Version | Notes |
|-----------|------------|---------|-------|
| Runtime | .NET | 10 | LTS |
| Framework | ASP.NET Core Minimal APIs | 10 | Performance |
| ORM | Entity Framework Core | 10 | + Dapper for perf |
| Validation | FluentValidation | Latest | Request validation |
| Mapping | Mapperly | Latest | Source-generated |
| Auth | Microsoft.Identity.Web | Latest | Azure AD B2C |
| API Docs | Swagger/OpenAPI | 3.0 | Via Swashbuckle |
| Testing | xUnit + NSubstitute | Latest | Unit/integration |

### 4.3 Machine Learning

| Component | Technology | Notes |
|-----------|------------|-------|
| Training | Python 3.11+ | scikit-learn, PyTorch, XGBoost |
| Feature Engineering | pandas, numpy | Data manipulation |
| Time Series | Prophet, statsmodels | ARIMA, seasonal |
| Deep Learning | PyTorch | LSTM, Transformer |
| Inference (.NET) | ML.NET 3.0 | ONNX model runtime |
| Experiment Tracking | MLflow | Optional |

### 4.4 Infrastructure

| Component | Technology | Notes |
|-----------|------------|-------|
| Frontend Hosting | Azure Static Web Apps | Free tier available |
| API Hosting | Azure App Service | B1 (dev) / P1v3 (prod) |
| Functions | Azure Functions | Consumption plan |
| Containers | Azure Container Apps | For ML workloads |
| SQL Database | Azure SQL Serverless | Auto-pause |
| NoSQL Database | Cosmos DB | Serverless + autoscale |
| Cache | Azure Cache for Redis | Basic C0 |
| Storage | Azure Blob Storage | Hot/Cool/Archive |
| Messaging | Azure Service Bus | Basic tier |
| CDN | Azure Front Door | Standard tier |
| Auth | Azure AD B2C | Free up to 50k MAU |
| Monitoring | Application Insights | Standard |
| CI/CD | GitHub Actions | Free |
| IaC | Bicep / Terraform | Infrastructure as Code |

---

## 5. Data Layer

### 5.1 Azure SQL Schema

```sql
-- Users and Authentication
CREATE TABLE Users (
    Id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    AzureAdB2CId NVARCHAR(100) NOT NULL UNIQUE,
    Email NVARCHAR(255) NOT NULL UNIQUE,
    DisplayName NVARCHAR(100),
    Tier NVARCHAR(20) NOT NULL DEFAULT 'Free', -- Free, Premium, Pro
    CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    UpdatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    LastLoginAt DATETIME2
);

-- Watchlists
CREATE TABLE Watchlists (
    Id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    UserId UNIQUEIDENTIFIER NOT NULL REFERENCES Users(Id),
    Name NVARCHAR(100) NOT NULL,
    IsDefault BIT NOT NULL DEFAULT 0,
    CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    UpdatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE()
);

CREATE TABLE WatchlistItems (
    Id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    WatchlistId UNIQUEIDENTIFIER NOT NULL REFERENCES Watchlists(Id),
    Symbol NVARCHAR(10) NOT NULL,
    AddedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    Notes NVARCHAR(500),
    UNIQUE(WatchlistId, Symbol)
);

-- Alerts
CREATE TABLE Alerts (
    Id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    UserId UNIQUEIDENTIFIER NOT NULL REFERENCES Users(Id),
    Symbol NVARCHAR(10) NOT NULL,
    AlertType NVARCHAR(50) NOT NULL, -- PriceAbove, PriceBelow, PatternDetected, PredictionChange
    Condition NVARCHAR(MAX) NOT NULL, -- JSON condition
    IsActive BIT NOT NULL DEFAULT 1,
    LastTriggeredAt DATETIME2,
    CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE()
);

CREATE TABLE AlertHistory (
    Id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    AlertId UNIQUEIDENTIFIER NOT NULL REFERENCES Alerts(Id),
    TriggeredAt DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    Message NVARCHAR(500),
    DeliveryChannel NVARCHAR(20), -- Email, Push, Webhook
    DeliveryStatus NVARCHAR(20) -- Sent, Failed, Pending
);

-- Subscriptions (Stripe integration)
CREATE TABLE Subscriptions (
    Id UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    UserId UNIQUEIDENTIFIER NOT NULL REFERENCES Users(Id),
    StripeCustomerId NVARCHAR(100),
    StripeSubscriptionId NVARCHAR(100),
    PlanId NVARCHAR(50) NOT NULL,
    Status NVARCHAR(20) NOT NULL, -- Active, Canceled, PastDue
    CurrentPeriodStart DATETIME2,
    CurrentPeriodEnd DATETIME2,
    CreatedAt DATETIME2 NOT NULL DEFAULT GETUTCDATE()
);

-- Indexes
CREATE INDEX IX_Users_AzureAdB2CId ON Users(AzureAdB2CId);
CREATE INDEX IX_Watchlists_UserId ON Watchlists(UserId);
CREATE INDEX IX_Alerts_UserId_Symbol ON Alerts(UserId, Symbol);
```

### 5.2 Cosmos DB Collections

#### Container: `prices`
```json
{
  "id": "AAPL_2026-02-02",
  "symbol": "AAPL",
  "date": "2026-02-02",
  "open": 185.50,
  "high": 187.25,
  "low": 184.80,
  "close": 186.90,
  "adjustedClose": 186.90,
  "volume": 52340000,
  "source": "polygon",
  "fetchedAt": "2026-02-02T21:35:00Z",
  "_partitionKey": "AAPL"
}
```
- **Partition Key:** `/symbol`
- **TTL:** None (permanent storage)
- **RU/s:** 400-4000 autoscale

#### Container: `predictions`
```json
{
  "id": "AAPL_2026-02-02_short",
  "symbol": "AAPL",
  "date": "2026-02-02",
  "horizon": "short",
  "prediction": {
    "direction": "bullish",
    "targetPrice": 192.50,
    "confidence": 0.73,
    "range": {
      "low": 183.00,
      "high": 195.00
    }
  },
  "modelVersion": "v1.2.0",
  "features": {
    "rsi": 58.3,
    "macdSignal": "bullish",
    "sma20": 184.20
  },
  "createdAt": "2026-02-02T21:45:00Z",
  "_partitionKey": "AAPL"
}
```
- **Partition Key:** `/symbol`
- **TTL:** 90 days (auto-cleanup old predictions)
- **RU/s:** 400-2000 autoscale

#### Container: `patterns`
```json
{
  "id": "AAPL_2026-02-02_hammer",
  "symbol": "AAPL",
  "date": "2026-02-02",
  "patternType": "candlestick",
  "patternName": "hammer",
  "signal": "bullish",
  "strength": 0.85,
  "description": "Hammer pattern detected, indicating potential reversal",
  "priceAtDetection": 186.90,
  "createdAt": "2026-02-02T21:50:00Z",
  "_partitionKey": "AAPL"
}
```
- **Partition Key:** `/symbol`
- **TTL:** 30 days
- **RU/s:** 400-1000 autoscale

#### Container: `symbols`
```json
{
  "id": "AAPL",
  "symbol": "AAPL",
  "name": "Apple Inc.",
  "exchange": "NASDAQ",
  "sector": "Technology",
  "industry": "Consumer Electronics",
  "marketCap": 2850000000000,
  "isActive": true,
  "lastUpdated": "2026-02-02T21:30:00Z",
  "_partitionKey": "NASDAQ"
}
```
- **Partition Key:** `/exchange`
- **TTL:** None
- **RU/s:** 400 fixed

---

## 6. API Specification

### 6.1 REST Endpoints

#### Authentication
```
POST   /api/auth/register          - Register new user (via B2C redirect)
POST   /api/auth/login             - Login (via B2C redirect)
POST   /api/auth/logout            - Logout
GET    /api/auth/me                - Get current user profile
```

#### Stocks
```
GET    /api/stocks                 - List stocks (paginated, searchable)
GET    /api/stocks/{symbol}        - Get stock details
GET    /api/stocks/{symbol}/prices - Get price history
GET    /api/stocks/{symbol}/prices/latest - Get latest price
GET    /api/stocks/{symbol}/indicators - Get technical indicators
```

#### Predictions
```
GET    /api/predictions/{symbol}           - Get all predictions for symbol
GET    /api/predictions/{symbol}/{horizon} - Get specific horizon prediction
GET    /api/predictions/top                - Get top predicted stocks
```

#### Patterns
```
GET    /api/patterns/{symbol}      - Get detected patterns for symbol
GET    /api/patterns/latest        - Get latest patterns across all stocks
GET    /api/patterns/types         - Get list of pattern types
```

#### Watchlists
```
GET    /api/watchlists             - Get user's watchlists
POST   /api/watchlists             - Create watchlist
GET    /api/watchlists/{id}        - Get watchlist details
PUT    /api/watchlists/{id}        - Update watchlist
DELETE /api/watchlists/{id}        - Delete watchlist
POST   /api/watchlists/{id}/items  - Add stock to watchlist
DELETE /api/watchlists/{id}/items/{symbol} - Remove stock
```

#### Alerts
```
GET    /api/alerts                 - Get user's alerts
POST   /api/alerts                 - Create alert
GET    /api/alerts/{id}            - Get alert details
PUT    /api/alerts/{id}            - Update alert
DELETE /api/alerts/{id}            - Delete alert
GET    /api/alerts/history         - Get alert history
```

#### User
```
GET    /api/user/profile           - Get profile
PUT    /api/user/profile           - Update profile
GET    /api/user/subscription      - Get subscription status
POST   /api/user/subscription      - Create/update subscription
```

### 6.2 Response Format

```json
{
  "success": true,
  "data": { },
  "meta": {
    "page": 1,
    "pageSize": 20,
    "totalCount": 150,
    "totalPages": 8
  },
  "errors": []
}
```

### 6.3 Error Response

```json
{
  "success": false,
  "data": null,
  "errors": [
    {
      "code": "VALIDATION_ERROR",
      "message": "Symbol is required",
      "field": "symbol"
    }
  ]
}
```

---

## 7. Frontend Specification

### 7.1 Page Structure

```
/                           - Landing page (marketing)
/dashboard                  - Main dashboard (authenticated)
/stocks                     - Stock screener/list
/stocks/:symbol             - Individual stock detail
/stocks/:symbol/chart       - Full chart view
/watchlists                 - Manage watchlists
/watchlists/:id             - Watchlist detail
/alerts                     - Manage alerts
/predictions                - Prediction overview
/account                    - Account settings
/account/subscription       - Subscription management
```

### 7.2 Component Hierarchy

```
App
├── Layout
│   ├── Header
│   │   ├── Logo
│   │   ├── Navigation
│   │   ├── SearchBar
│   │   └── UserMenu
│   ├── Sidebar (collapsible)
│   │   ├── WatchlistQuick
│   │   └── MarketSummary
│   └── Footer
│
├── Pages
│   ├── Dashboard
│   │   ├── MarketOverview
│   │   ├── TopPredictions
│   │   ├── WatchlistSummary
│   │   └── RecentAlerts
│   │
│   ├── StockDetail
│   │   ├── StockHeader
│   │   ├── PriceChart (TradingView)
│   │   ├── PredictionCard
│   │   ├── PatternsList
│   │   ├── IndicatorsTable
│   │   └── StockActions
│   │
│   └── ...
│
└── Shared
    ├── StockCard
    ├── PredictionBadge
    ├── PatternChip
    ├── PriceDisplay
    ├── LoadingSpinner
    └── ErrorBoundary
```

### 7.3 State Management

```typescript
// stores/stockStore.ts
interface StockStore {
  // Data
  stocks: Map<string, Stock>;
  prices: Map<string, PriceData[]>;
  predictions: Map<string, Prediction>;
  patterns: Map<string, Pattern[]>;

  // UI State
  selectedSymbol: string | null;
  chartTimeframe: '1D' | '1W' | '1M' | '3M' | '1Y' | '5Y';

  // Actions
  fetchStock: (symbol: string) => Promise<void>;
  fetchPrices: (symbol: string, range: DateRange) => Promise<void>;
  setSelectedSymbol: (symbol: string) => void;
}

// stores/userStore.ts
interface UserStore {
  user: User | null;
  watchlists: Watchlist[];
  alerts: Alert[];
  subscription: Subscription | null;

  // Actions
  fetchUser: () => Promise<void>;
  updateProfile: (data: ProfileUpdate) => Promise<void>;
  addToWatchlist: (listId: string, symbol: string) => Promise<void>;
}
```

---

## 8. ML/AI Pipeline

### 8.1 Model Architecture

#### Short-Term Model (1-5 days)
```
Input Features:
├── Price: Last 60 days OHLCV
├── Technical: RSI, MACD, SMA(10,20,50), EMA(12,26), Bollinger Bands
├── Volume: Volume MA, OBV, Volume Profile
└── Momentum: ROC, Stochastic, Williams %R

Model: XGBoost + LSTM Ensemble
├── XGBoost: Feature-based classification (up/down/neutral)
├── LSTM: Sequence learning (60-day window)
└── Ensemble: Weighted average based on recent accuracy

Output:
├── Direction: bullish/bearish/neutral
├── Confidence: 0-1 score
├── Price Range: (low, target, high)
└── Key Factors: Top 3 contributing features
```

#### Medium-Term Model (1-4 weeks)
```
Input Features:
├── All short-term features
├── Fundamentals: P/E, P/B, Market Cap, Revenue Growth
├── Sentiment: News sentiment score (optional Phase 2)
└── Sector: Sector performance correlation

Model: Temporal Fusion Transformer (TFT)
├── Multi-horizon prediction
├── Attention-based feature importance
└── Uncertainty quantification

Output:
├── Weekly predictions for next 4 weeks
├── Confidence intervals
└── Trend classification
```

#### Long-Term Model (1-6 months)
```
Input Features:
├── Monthly aggregated prices
├── Fundamental ratios
├── Macro indicators (optional)
└── Seasonality patterns

Model: Prophet + Gradient Boosting
├── Prophet: Trend + seasonality decomposition
├── XGBoost: Residual prediction
└── Combined: Final forecast

Output:
├── Monthly price targets
├── Trend direction
└── Risk assessment
```

### 8.2 Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ML TRAINING PIPELINE                                 │
│                   (Weekly on Sundays)                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. DATA EXTRACTION                                                      │
│     └── Extract last 5 years of price data from Cosmos DB               │
│                                                                          │
│  2. FEATURE ENGINEERING                                                  │
│     ├── Calculate all technical indicators                              │
│     ├── Create lagged features                                          │
│     ├── Normalize/scale features                                        │
│     └── Train/validation/test split (70/15/15)                          │
│                                                                          │
│  3. MODEL TRAINING                                                       │
│     ├── Train short-term models                                         │
│     ├── Train medium-term models                                        │
│     ├── Train long-term models                                          │
│     └── Hyperparameter tuning (Optuna)                                  │
│                                                                          │
│  4. EVALUATION                                                           │
│     ├── Calculate accuracy metrics (Precision, Recall, F1)              │
│     ├── Compare against baseline                                        │
│     ├── Backtest predictions                                            │
│     └── Generate evaluation report                                      │
│                                                                          │
│  5. MODEL EXPORT                                                         │
│     ├── Export to ONNX format (for ML.NET)                              │
│     ├── Version with timestamp                                          │
│     └── Upload to Blob Storage                                          │
│                                                                          │
│  6. DEPLOYMENT                                                           │
│     ├── Update model reference in config                                │
│     ├── Warm up inference containers                                    │
│     └── Validate predictions on sample data                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.3 Inference Pipeline

```csharp
// Simplified ML.NET inference example
public class PredictionService
{
    private readonly MLContext _mlContext;
    private readonly ITransformer _model;

    public async Task<Prediction> PredictAsync(string symbol)
    {
        // 1. Fetch features from cache/database
        var features = await _featureService.GetFeaturesAsync(symbol);

        // 2. Create prediction engine
        var predictionEngine = _mlContext.Model
            .CreatePredictionEngine<StockFeatures, StockPrediction>(_model);

        // 3. Make prediction
        var result = predictionEngine.Predict(features);

        // 4. Post-process and return
        return new Prediction
        {
            Symbol = symbol,
            Direction = result.PredictedLabel,
            Confidence = result.Score.Max(),
            TargetPrice = CalculateTargetPrice(features.CurrentPrice, result)
        };
    }
}
```

---

## 9. Background Jobs

### 9.1 Job Schedule

| Job | Schedule | Runtime | Description |
|-----|----------|---------|-------------|
| FetchDailyPrices | 4:30 PM ET (M-F) | ~15 min | Fetch EOD prices for all symbols |
| CalculateIndicators | After FetchDailyPrices | ~5 min | Calculate technical indicators |
| RunPredictions | After CalculateIndicators | ~10 min | Run ML inference |
| DetectPatterns | After RunPredictions | ~5 min | Pattern detection |
| ProcessAlerts | After DetectPatterns | ~2 min | Match and send alerts |
| WeeklyModelTraining | Sunday 2:00 AM | ~2 hours | Retrain ML models |
| DataCleanup | Daily 3:00 AM | ~5 min | Clean expired data |
| HealthCheck | Every 5 minutes | ~10 sec | System health monitoring |

### 9.2 Durable Functions Orchestration

```csharp
[FunctionName("DailyPipelineOrchestrator")]
public static async Task RunOrchestrator(
    [OrchestrationTrigger] IDurableOrchestrationContext context)
{
    var symbols = await context.CallActivityAsync<List<string>>("GetActiveSymbols");

    // Step 1: Fetch prices in parallel batches
    var fetchTasks = symbols
        .Chunk(100)
        .Select(batch => context.CallActivityAsync("FetchPrices", batch));
    await Task.WhenAll(fetchTasks);

    // Step 2: Calculate indicators
    await context.CallActivityAsync("CalculateIndicators", symbols);

    // Step 3: Run predictions
    await context.CallActivityAsync("RunPredictions", symbols);

    // Step 4: Detect patterns
    await context.CallActivityAsync("DetectPatterns", symbols);

    // Step 5: Process alerts
    await context.CallActivityAsync("ProcessAlerts", null);

    // Log completion
    await context.CallActivityAsync("LogPipelineComplete", context.InstanceId);
}
```

---

## 10. Security

### 10.1 Authentication & Authorization

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    AUTHENTICATION FLOW                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. User clicks "Sign In"                                                │
│  2. Redirect to Azure AD B2C                                             │
│  3. User authenticates (email/password or social)                        │
│  4. B2C issues JWT token                                                 │
│  5. Frontend stores token (httpOnly cookie or secure storage)            │
│  6. API validates token on each request                                  │
│  7. Claims extracted for authorization                                   │
│                                                                          │
│  Token Claims:                                                           │
│  ├── sub: User ID                                                        │
│  ├── email: User email                                                   │
│  ├── tier: Free/Premium/Pro                                              │
│  └── roles: user/admin                                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Security Measures

| Layer | Measure |
|-------|---------|
| Network | Azure Front Door WAF, DDoS protection |
| Transport | TLS 1.3, HTTPS only |
| API | Rate limiting, input validation, CORS |
| Auth | Azure AD B2C, MFA support, JWT validation |
| Data | Encryption at rest (Azure-managed keys) |
| Secrets | Azure Key Vault |
| Logging | No PII in logs, audit trails |

### 10.3 Rate Limits

| Tier | Requests/minute | Requests/day |
|------|-----------------|--------------|
| Free | 20 | 500 |
| Premium | 60 | 5,000 |

---

## 11. Infrastructure

### 11.1 Environment Strategy

| Environment | Purpose | Scale | Data |
|-------------|---------|-------|------|
| Development | Local dev, debugging | Minimal | Sample data |
| Staging | Integration testing | 1x prod | Anonymized copy |
| Production | Live users | Auto-scale | Real data |

### 11.2 Azure Resource Naming

```
Pattern: {project}-{environment}-{resource}-{region}

Examples:
- pricepred-prod-sql-eastus
- pricepred-prod-cosmos-eastus
- pricepred-prod-app-eastus
- pricepred-staging-func-eastus
```

### 11.3 Bicep Infrastructure (Simplified)

```bicep
// main.bicep
param environment string
param location string = 'eastus'

// App Service Plan
resource appServicePlan 'Microsoft.Web/serverfarms@2022-03-01' = {
  name: 'pricepred-${environment}-plan-${location}'
  location: location
  sku: {
    name: environment == 'prod' ? 'P1v3' : 'B1'
  }
}

// API App Service
resource apiApp 'Microsoft.Web/sites@2022-03-01' = {
  name: 'pricepred-${environment}-api-${location}'
  location: location
  properties: {
    serverFarmId: appServicePlan.id
    httpsOnly: true
  }
}

// Cosmos DB
resource cosmosAccount 'Microsoft.DocumentDB/databaseAccounts@2022-05-15' = {
  name: 'pricepred-${environment}-cosmos-${location}'
  location: location
  properties: {
    databaseAccountOfferType: 'Standard'
    capabilities: [
      { name: 'EnableServerless' }
    ]
  }
}

// Azure SQL
resource sqlServer 'Microsoft.Sql/servers@2022-05-01-preview' = {
  name: 'pricepred-${environment}-sql-${location}'
  location: location
  properties: {
    administratorLogin: 'sqladmin'
    administratorLoginPassword: sqlAdminPassword
  }
}
```

---

## 12. Cost Estimates

### 12.1 Development Environment

| Service | Configuration | Monthly Cost |
|---------|---------------|--------------|
| App Service | B1 (1 core, 1.75 GB) | $13 |
| Azure SQL | Serverless (1 vCore, auto-pause) | $5-15 |
| Cosmos DB | Serverless (pay per RU) | $5-20 |
| Functions | Consumption (1M free) | $0 |
| Storage | 10 GB Hot | $2 |
| Redis | Basic C0 (250 MB) | $16 |
| Static Web Apps | Free tier | $0 |
| **Total** | | **$41-66/mo** |

### 12.2 Production Environment

| Service | Configuration | Monthly Cost |
|---------|---------------|--------------|
| App Service | P1v3 (2 core, 8 GB) | $138 |
| Azure SQL | S2 (50 DTU) | $75 |
| Cosmos DB | Autoscale 400-4000 RU | $50-200 |
| Functions | Consumption | $10-30 |
| Container Apps | 1 vCPU, 2 GB | $40-80 |
| Storage | 100 GB Hot + Archive | $10 |
| Redis | Standard C1 (1 GB) | $41 |
| Front Door | Standard | $35 |
| Static Web Apps | Standard | $9 |
| Service Bus | Basic | $10 |
| Key Vault | Standard | $3 |
| App Insights | 5 GB/mo | $12 |
| **Total** | | **$433-643/mo** |

### 12.3 Cost Optimization Tips

1. **Reserved Instances:** 1-year reservation saves ~35%
2. **Azure Hybrid Benefit:** Use existing licenses
3. **Auto-scaling:** Scale down during off-hours
4. **Data Tiering:** Move old data to Cool/Archive
5. **Cosmos DB TTL:** Auto-delete old predictions
6. **SQL Auto-pause:** Serverless pauses when idle
7. **Spot Instances:** For ML training containers

---

## 13. Development Phases

### Phase 1: Foundation (Weeks 1-4)

**Goals:** Basic infrastructure and data pipeline

- [ ] Set up Azure resources (SQL, Cosmos, App Service)
- [ ] Implement Azure AD B2C authentication
- [ ] Create .NET API project structure
- [ ] Set up React frontend with routing
- [ ] Implement user registration/login flow
- [ ] Create basic stock listing page
- [ ] Set up CI/CD pipeline

**Deliverables:**
- Users can register and log in
- Basic stock search functionality
- Infrastructure deployed

### Phase 2: Data Pipeline (Weeks 5-8)

**Goals:** Price data ingestion and storage

- [ ] Integrate Alpha Vantage API
- [ ] Create Azure Functions for data fetching
- [ ] Implement Cosmos DB data layer
- [ ] Build price history storage
- [ ] Add technical indicator calculations
- [ ] Create basic charting with TradingView
- [ ] Implement watchlist functionality

**Deliverables:**
- Daily price data for 100 pilot stocks
- Interactive price charts
- Working watchlist feature

### Phase 3: Predictions (Weeks 9-12)

**Goals:** ML model development and integration

- [ ] Develop XGBoost baseline model
- [ ] Create feature engineering pipeline
- [ ] Implement ML.NET inference
- [ ] Build prediction display UI
- [ ] Add confidence scoring
- [ ] Create prediction history tracking
- [ ] Expand to full stock universe

**Deliverables:**
- Working predictions for all US stocks
- Prediction accuracy dashboard
- Historical prediction tracking

### Phase 4: Patterns & Alerts (Weeks 13-16)

**Goals:** Pattern detection and user notifications

- [ ] Implement candlestick pattern detection
- [ ] Add chart pattern recognition
- [ ] Build alert system
- [ ] Integrate email notifications
- [ ] Create alert management UI
- [ ] Add pattern education content

**Deliverables:**
- Pattern detection for common patterns
- Email alert notifications
- Pattern education tooltips

### Phase 5: Premium Features (Weeks 17-20)

**Goals:** Monetization and advanced features

- [ ] Implement Stripe subscription integration
- [ ] Add premium feature gating
- [ ] Build custom screener
- [ ] Add LSTM/Transformer models
- [ ] Create API access for Pro tier
- [ ] Performance optimization

**Deliverables:**
- Working subscription system
- Premium feature differentiation
- API documentation

### Phase 6: Polish & Launch (Weeks 21-24)

**Goals:** Production readiness

- [ ] Security audit and penetration testing
- [ ] Performance testing and optimization
- [ ] Error handling and edge cases
- [ ] User documentation
- [ ] Marketing site
- [ ] Soft launch with beta users
- [ ] Monitor and iterate

**Deliverables:**
- Production-ready application
- Documentation complete
- Beta user feedback incorporated

---

## 14. Future Expansion

### 14.1 Market Adapters Architecture

```csharp
public interface IMarketDataProvider
{
    string MarketName { get; }
    Task<IEnumerable<Symbol>> GetSymbolsAsync();
    Task<PriceData> GetPriceAsync(string symbol);
    Task<IEnumerable<PriceData>> GetHistoricalPricesAsync(string symbol, DateRange range);
}

// Implementations
public class StockMarketProvider : IMarketDataProvider { }
public class CryptoMarketProvider : IMarketDataProvider { } // Future
public class CommoditiesProvider : IMarketDataProvider { } // Future
public class ForexProvider : IMarketDataProvider { } // Future
```

### 14.2 Expansion Roadmap

| Market | Data Sources | Est. Timeline |
|--------|--------------|---------------|
| Crypto | CoinGecko, Binance API | Q3 2026 |
| Commodities | Quandl, Alpha Vantage | Q4 2026 |
| Forex | OANDA, Alpha Vantage | Q4 2026 |
| International Stocks | IEX, Yahoo | Q1 2027 |
| Real Estate | Zillow API, ATTOM | Q2 2027 |

### 14.3 Feature Roadmap

| Feature | Description | Est. Timeline |
|---------|-------------|---------------|
| Mobile App | React Native iOS/Android | Q3 2026 |
| Social Features | Share predictions, follow traders | Q4 2026 |
| Paper Trading | Simulated trading with predictions | Q4 2026 |
| Backtesting | Test strategies on historical data | Q1 2027 |
| AI Chat | Natural language stock queries | Q2 2027 |
| Webhook Integrations | IFTTT, Zapier, Discord | Q2 2027 |

---

## Appendix

### A. Glossary

| Term | Definition |
|------|------------|
| EOD | End of Day (market close) |
| OHLCV | Open, High, Low, Close, Volume |
| RSI | Relative Strength Index |
| MACD | Moving Average Convergence Divergence |
| SMA | Simple Moving Average |
| EMA | Exponential Moving Average |
| RU | Request Unit (Cosmos DB throughput) |
| TTL | Time To Live (auto-deletion) |

### B. References

- [Azure Well-Architected Framework](https://docs.microsoft.com/azure/architecture/framework/)
- [TradingView Lightweight Charts](https://tradingview.github.io/lightweight-charts/)
- [ML.NET Documentation](https://docs.microsoft.com/dotnet/machine-learning/)
- [Polygon.io API](https://polygon.io/docs/)
- [Alpha Vantage API](https://www.alphavantage.co/documentation/)

---

*Document Version: 1.0*
*Last Updated: 2026-02-02*
*Status: Draft - Pending Review*
