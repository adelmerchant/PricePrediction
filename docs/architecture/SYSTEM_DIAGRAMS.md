# System Architecture Diagrams

## Table of Contents
1. [High-Level System Architecture](#high-level-system-architecture)
2. [Data Flow Architecture](#data-flow-architecture)
3. [ML Pipeline Architecture](#ml-pipeline-architecture)
4. [Component Interaction Diagram](#component-interaction-diagram)
5. [Deployment Topology](#deployment-topology)
6. [Database Entity Relationships](#database-entity-relationships)

---

## High-Level System Architecture

```mermaid
graph TB
    subgraph "Presentation Layer"
        UI[Blazor Server UI<br/>Port 5000]
        API[REST API<br/>Port 5001]
        SignalR[SignalR Hub<br/>Real-time]
    end

    subgraph "Application Services"
        TrendSvc[Trend Detection<br/>Service]
        PatternSvc[Pattern Detection<br/>Service]
        PredictionSvc[Price Prediction<br/>Service]
        AlertSvc[Alert Processor<br/>Service]
        BacktestSvc[Backtesting<br/>Engine]
    end

    subgraph "ML Engine"
        MLPipeline[ML Pipeline<br/>GPU-Accelerated]
        FeatureEng[Feature<br/>Engineering]
        ModelServing[Model Serving<br/>ONNX Runtime]
        ModelTraining[Model Training<br/>TorchSharp/CUDA]
    end

    subgraph "Data Layer"
        SQLRepo[SQL Server<br/>Repository]
        MongoRepo[MongoDB<br/>Repository]
        RedisCache[Redis Cache<br/>Service]
    end

    subgraph "Background Jobs"
        DataFetcher[Data Fetcher<br/>Quartz.NET]
        ModelTrainer[Model Trainer<br/>Scheduled Jobs]
        AlertMonitor[Alert Monitor<br/>Hangfire]
    end

    subgraph "Data Storage"
        SQLDB[(SQL Server<br/>Relational Data)]
        MongoDB[(MongoDB<br/>Document Store)]
        Redis[(Redis<br/>In-Memory Cache)]
    end

    subgraph "External Services"
        Polygon[Polygon.io<br/>Market Data]
        AlphaVantage[Alpha Vantage<br/>Backup Data]
        FMP[Financial Modeling Prep<br/>Fundamentals]
        NewsAPI[NewsAPI<br/>Sentiment]
    end

    UI --> API
    UI --> SignalR
    API --> TrendSvc
    API --> PatternSvc
    API --> PredictionSvc
    API --> AlertSvc
    API --> BacktestSvc

    TrendSvc --> FeatureEng
    PatternSvc --> MLPipeline
    PredictionSvc --> ModelServing
    BacktestSvc --> MLPipeline

    FeatureEng --> SQLRepo
    FeatureEng --> MongoRepo
    ModelServing --> RedisCache

    TrendSvc --> RedisCache
    PatternSvc --> RedisCache
    PredictionSvc --> RedisCache

    SQLRepo --> SQLDB
    MongoRepo --> MongoDB
    RedisCache --> Redis

    DataFetcher --> Polygon
    DataFetcher --> AlphaVantage
    DataFetcher --> FMP
    DataFetcher --> NewsAPI

    DataFetcher --> MongoRepo
    DataFetcher --> SQLRepo

    ModelTrainer --> ModelTraining
    ModelTraining --> MongoRepo

    AlertMonitor --> AlertSvc
    AlertSvc --> SignalR

    style MLPipeline fill:#90EE90
    style ModelTraining fill:#90EE90
    style Redis fill:#FFB6C1
    style SQLDB fill:#87CEEB
    style MongoDB fill:#98FB98
```

---

## Data Flow Architecture

### Real-Time Quote Processing Flow

```mermaid
sequenceDiagram
    participant ExtAPI as External API<br/>(Polygon.io)
    participant Fetcher as Data Fetcher<br/>Service
    participant Validator as Data Validator
    participant Mongo as MongoDB<br/>(Raw Data)
    participant SQL as SQL Server<br/>(OHLCV)
    participant Redis as Redis Cache
    participant SignalR as SignalR Hub
    participant UI as Blazor UI

    loop Every 1-5 minutes (Market Hours)
        Fetcher->>ExtAPI: GET /quotes (symbols)
        ExtAPI-->>Fetcher: JSON Response
        Fetcher->>Validator: Validate Data
        Validator->>Validator: Check schema, outliers

        alt Data Valid
            Validator->>Mongo: Store Raw JSON
            Validator->>SQL: Insert/Update OHLCV
            Validator->>Redis: Cache (TTL 5min)
            Redis->>SignalR: Publish Update Event
            SignalR->>UI: Push Real-time Quote
        else Data Invalid
            Validator->>Fetcher: Log Error
            Fetcher->>ExtAPI: Retry with backoff
        end
    end
```

### ML Prediction Pipeline Flow

```mermaid
flowchart LR
    subgraph "Data Preparation"
        A[Historical Data<br/>SQL Server] --> B[Feature Engineering]
        B --> C[Technical Indicators<br/>SMA, RSI, MACD]
        B --> D[Price Features<br/>Returns, Volatility]
        B --> E[Volume Features<br/>OBV, VWAP]
    end

    subgraph "Feature Store"
        C --> F[Feature Vector<br/>Redis Cache]
        D --> F
        E --> F
    end

    subgraph "ML Inference"
        F --> G{Model Type}
        G -->|Intra-day| H[LSTM Model<br/>ONNX Runtime GPU]
        G -->|Short-term| I[GRU + XGBoost<br/>Ensemble]
        G -->|Long-term| J[Prophet +<br/>Fundamentals]
    end

    subgraph "Post-Processing"
        H --> K[Ensemble Aggregator]
        I --> K
        J --> K
        K --> L[Confidence Calculator]
        L --> M[Risk Assessment]
    end

    subgraph "Output"
        M --> N[Price Predictions<br/>SQL Server]
        N --> O[Cache Results<br/>Redis TTL 1hr]
        O --> P[API Response]
    end

    style F fill:#FFD700
    style H fill:#90EE90
    style I fill:#90EE90
    style J fill:#90EE90
    style O fill:#FFB6C1
```

### Backtesting Workflow

```mermaid
flowchart TD
    A[Start Backtest] --> B[Load Historical Data<br/>SQL Server]
    B --> C[Define Date Ranges<br/>Train/Val/Test]
    C --> D[Walk-Forward Loop]

    D --> E[Train Model<br/>Training Window]
    E --> F[Validate Hyperparameters<br/>Validation Window]
    F --> G[Generate Predictions<br/>Test Window]

    G --> H[Simulate Trades]
    H --> I[Apply Transaction Costs<br/>Slippage, Commission]
    I --> J[Calculate P&L]

    J --> K{More Windows?}
    K -->|Yes| L[Roll Window Forward]
    L --> D
    K -->|No| M[Aggregate Results]

    M --> N[Calculate Metrics<br/>Sharpe, Max DD, Win Rate]
    N --> O[Generate Report<br/>Save to MongoDB]
    O --> P[Visualize Results<br/>Equity Curve, Stats]
    P --> Q[End]

    style E fill:#90EE90
    style H fill:#FFE4B5
    style N fill:#87CEEB
    style O fill:#DDA0DD
```

---

## ML Pipeline Architecture

### Model Training Pipeline

```mermaid
graph TB
    subgraph "Data Acquisition"
        A[Historical OHLCV<br/>SQL Server] --> B[Data Splitter]
        B --> C[Training Set<br/>70%]
        B --> D[Validation Set<br/>15%]
        B --> E[Test Set<br/>15%]
    end

    subgraph "Feature Engineering"
        C --> F[Feature Extractor]
        D --> F
        E --> F
        F --> G[Normalization<br/>StandardScaler]
        G --> H[Feature Store<br/>MongoDB]
    end

    subgraph "Model Training - GPU Accelerated"
        H --> I{Model Selection}
        I --> J[LSTM/GRU<br/>TorchSharp CUDA]
        I --> K[Transformer<br/>TorchSharp CUDA]
        I --> L[XGBoost/LightGBM<br/>GPU Histogram]

        J --> M[Hyperparameter Tuning<br/>Optuna.NET]
        K --> M
        L --> M
    end

    subgraph "Model Validation"
        M --> N[Cross-Validation<br/>Time Series CV]
        N --> O[Performance Metrics<br/>MAE, RMSE, MAPE]
        O --> P{Meets Threshold?}
        P -->|No| Q[Adjust Hyperparameters]
        Q --> M
        P -->|Yes| R[Model Accepted]
    end

    subgraph "Model Deployment"
        R --> S[Export to ONNX<br/>GPU-optimized]
        S --> T[Version Control<br/>MongoDB + Blob]
        T --> U[Load to Inference<br/>ONNX Runtime]
    end

    subgraph "Monitoring"
        U --> V[Prediction Tracking]
        V --> W[Model Drift Detection]
        W --> X{Drift Detected?}
        X -->|Yes| Y[Trigger Retraining]
        Y --> A
        X -->|No| V
    end

    style J fill:#90EE90
    style K fill:#90EE90
    style L fill:#90EE90
    style S fill:#FFD700
    style U fill:#87CEEB
```

### Inference Pipeline

```mermaid
sequenceDiagram
    participant API as REST API
    participant Cache as Redis Cache
    participant FeatureEng as Feature Engineering
    participant ModelServ as Model Serving<br/>ONNX Runtime
    participant DB as SQL Server
    participant User as UI Client

    User->>API: Request Prediction<br/>(symbol, timeframe)
    API->>Cache: Check Cache

    alt Cache Hit
        Cache-->>API: Return Cached Prediction
        API-->>User: Prediction Response
    else Cache Miss
        API->>DB: Get Recent OHLCV Data
        DB-->>API: Historical Bars
        API->>FeatureEng: Extract Features
        FeatureEng-->>API: Feature Vector

        API->>ModelServ: Inference Request
        Note over ModelServ: GPU-accelerated<br/>ONNX Runtime
        ModelServ-->>API: Prediction Output

        API->>Cache: Store (TTL 1hr)
        API->>DB: Save Prediction
        API-->>User: Prediction Response
    end
```

---

## Component Interaction Diagram

### Alert Processing System

```mermaid
graph TB
    subgraph "Data Sources"
        A[Real-time Quotes<br/>Redis Stream]
        B[Price Predictions<br/>SQL Server]
        C[Pattern Detections<br/>MongoDB]
    end

    subgraph "Alert Monitor Service"
        D[Event Listener<br/>Redis Pub/Sub]
        A --> D
        B --> D
        C --> D

        D --> E[Rule Engine<br/>NRules]
        E --> F{Condition Met?}
    end

    subgraph "Alert Rules"
        G[Price Target Rules]
        H[Pattern Rules]
        I[Volatility Rules]
        J[Custom User Rules]

        F --> G
        F --> H
        F --> I
        F --> J
    end

    subgraph "Alert Processor"
        G --> K[Alert Generator]
        H --> K
        I --> K
        J --> K

        K --> L[Priority Classifier<br/>High/Med/Low]
        L --> M[Deduplication<br/>5-min window]
    end

    subgraph "Notification Delivery"
        M --> N{Delivery Channel}
        N --> O[SignalR Hub<br/>Web Push]
        N --> P[Email<br/>MailKit]
        N --> Q[SMS<br/>Twilio]

        O --> R[Blazor UI]
        P --> S[User Email]
        Q --> T[User Phone]
    end

    subgraph "Alert History"
        M --> U[(SQL Server<br/>Alerts Table)]
        U --> V[Alert Analytics<br/>Dashboard]
    end

    style K fill:#FFB6C1
    style E fill:#FFD700
    style O fill:#90EE90
```

---

## Deployment Topology

### Phase 1: Local Workstation

```mermaid
graph TB
    subgraph "Workstation - Windows 11"
        subgraph "Application Layer"
            A[.NET 10 Web Host<br/>Kestrel Server]
            A --> B[Blazor UI :5000]
            A --> C[REST API :5001]
            A --> D[SignalR :5001]
            A --> E[Hangfire Dashboard :5002]
        end

        subgraph "Background Services"
            F[Data Fetcher<br/>Quartz.NET Scheduler]
            G[Model Trainer<br/>Scheduled Tasks]
            H[Alert Monitor<br/>Hangfire Worker]
        end

        subgraph "ML Runtime"
            I[TorchSharp + CUDA 12.x<br/>NVIDIA RTX 3090]
            J[ONNX Runtime GPU<br/>Inference Engine]
        end

        subgraph "Data Tier"
            K[(SQL Server 2022<br/>Port 1433)]
            L[(MongoDB 6.0<br/>Port 27017)]
            M[(Redis 7.x<br/>Port 6379)]
        end

        subgraph "Storage"
            N[NVMe SSD 1<br/>1TB - OS + DB]
            O[NVMe SSD 2<br/>2TB - Data + Models]
        end

        B --> C
        C --> F
        C --> I
        C --> J
        F --> K
        G --> I
        H --> D
        J --> M
        K --> N
        L --> O
        M --> N
    end

    subgraph "External APIs"
        P[Polygon.io]
        Q[Alpha Vantage]
        R[FMP API]
    end

    F --> P
    F --> Q
    F --> R

    subgraph "Monitoring"
        S[Seq Log Server<br/>Port 5341]
        T[Prometheus<br/>Port 9090]
        U[Grafana<br/>Port 3000]
    end

    A --> S
    F --> S
    T --> U

    style I fill:#90EE90
    style J fill:#90EE90
    style K fill:#87CEEB
    style L fill:#98FB98
    style M fill:#FFB6C1
```

### Phase 2: Hybrid Cloud Deployment

```mermaid
graph TB
    subgraph "Azure Cloud"
        subgraph "App Services"
            A[Azure App Service<br/>Blazor UI]
            B[Container Apps<br/>Background Jobs]
        end

        subgraph "API Layer"
            C[Azure API Management<br/>Rate Limiting + OAuth]
            A --> C
        end

        subgraph "Data Services"
            D[(Azure SQL Database<br/>Serverless)]
            E[(Cosmos DB<br/>MongoDB API)]
            F[(Azure Cache<br/>Redis Premium)]
        end

        subgraph "Storage"
            G[Blob Storage<br/>Model Artifacts]
            H[Queue Storage<br/>Job Queue]
        end

        subgraph "Monitoring"
            I[Application Insights<br/>Telemetry]
            J[Azure Monitor<br/>Metrics + Alerts]
        end

        C --> B
        B --> D
        B --> E
        B --> F
        B --> G
        B --> H
        A --> I
        B --> I
        I --> J
    end

    subgraph "Local Workstation"
        subgraph "ML Workload"
            K[Model Training<br/>TorchSharp CUDA]
            L[Model Inference<br/>ONNX Runtime GPU]
        end

        subgraph "Hardware"
            M[NVIDIA RTX 3090<br/>24GB VRAM]
        end

        K --> M
        L --> M
    end

    subgraph "Data Sync"
        N[Model Upload<br/>via Azure SDK]
        K --> N
        N --> G
        G --> L
    end

    subgraph "External Data"
        O[Polygon.io API]
        P[Financial Modeling Prep]
    end

    B --> O
    B --> P

    style K fill:#90EE90
    style L fill:#90EE90
    style M fill:#90EE90
    style D fill:#87CEEB
    style E fill:#98FB98
    style F fill:#FFB6C1
    style G fill:#FFD700
```

---

## Database Entity Relationships

### SQL Server Schema

```mermaid
erDiagram
    STOCKS ||--o{ OHLCV_DATA : has
    STOCKS ||--o{ PREDICTIONS : has
    STOCKS ||--o{ ALERTS : triggers
    STOCKS ||--o{ PATTERNS : contains
    STOCKS ||--o{ FUNDAMENTALS : has

    USERS ||--o{ ALERTS : configures
    USERS ||--o{ WATCHLISTS : creates
    WATCHLISTS ||--o{ WATCHLIST_ITEMS : contains
    WATCHLIST_ITEMS }o--|| STOCKS : references

    ML_MODELS ||--o{ PREDICTIONS : generates
    ML_MODELS ||--o{ BACKTESTS : used_in
    BACKTESTS ||--o{ BACKTEST_TRADES : contains

    STOCKS {
        int StockId PK
        string Symbol UK
        string CompanyName
        string Exchange
        string Sector
        string Industry
        decimal MarketCap
        datetime ListingDate
        boolean IsActive
        datetime CreatedAt
        datetime UpdatedAt
    }

    OHLCV_DATA {
        bigint OhlcvId PK
        int StockId FK
        datetime Timestamp
        decimal Open
        decimal High
        decimal Low
        decimal Close
        bigint Volume
        decimal VWAP
        int Timeframe
        datetime CreatedAt
    }

    PREDICTIONS {
        bigint PredictionId PK
        int StockId FK
        int ModelId FK
        datetime PredictedAt
        datetime TargetDate
        decimal PredictedPrice
        decimal ConfidenceLower
        decimal ConfidenceUpper
        decimal ConfidenceScore
        int Timeframe
        json FeatureImportance
        datetime CreatedAt
    }

    ALERTS {
        bigint AlertId PK
        int UserId FK
        int StockId FK
        string AlertType
        json RuleConfig
        string Status
        datetime TriggeredAt
        datetime SentAt
        string NotificationChannel
        string Priority
        datetime CreatedAt
        datetime UpdatedAt
    }

    PATTERNS {
        bigint PatternId PK
        int StockId FK
        string PatternType
        datetime DetectedAt
        decimal ConfidenceScore
        json PatternData
        string Timeframe
        datetime StartDate
        datetime EndDate
        datetime CreatedAt
    }

    ML_MODELS {
        int ModelId PK
        string ModelName
        string ModelType
        string Version
        json Hyperparameters
        decimal ValidationAccuracy
        decimal TestAccuracy
        datetime TrainedAt
        string Status
        string StoragePath
        datetime CreatedAt
    }

    BACKTESTS {
        bigint BacktestId PK
        int ModelId FK
        string BacktestName
        datetime StartDate
        datetime EndDate
        json Configuration
        decimal TotalReturn
        decimal SharpeRatio
        decimal MaxDrawdown
        decimal WinRate
        json PerformanceMetrics
        datetime CreatedAt
    }

    FUNDAMENTALS {
        bigint FundamentalId PK
        int StockId FK
        datetime ReportDate
        string Period
        decimal Revenue
        decimal NetIncome
        decimal EPS
        decimal PE_Ratio
        decimal PB_Ratio
        decimal ROE
        decimal DebtToEquity
        datetime CreatedAt
    }

    USERS {
        int UserId PK
        string Email UK
        string PasswordHash
        string FirstName
        string LastName
        json Preferences
        datetime CreatedAt
        datetime LastLoginAt
    }

    WATCHLISTS {
        int WatchlistId PK
        int UserId FK
        string Name
        string Description
        datetime CreatedAt
        datetime UpdatedAt
    }

    WATCHLIST_ITEMS {
        int WatchlistItemId PK
        int WatchlistId FK
        int StockId FK
        int SortOrder
        json CustomSettings
        datetime AddedAt
    }

    BACKTEST_TRADES {
        bigint TradeId PK
        bigint BacktestId FK
        int StockId FK
        datetime EntryDate
        decimal EntryPrice
        datetime ExitDate
        decimal ExitPrice
        int Quantity
        string TradeType
        decimal ProfitLoss
        decimal Commission
        decimal Slippage
    }
```

### MongoDB Schema Design

```mermaid
graph TB
    subgraph "MongoDB Collections"
        A[raw_market_data]
        B[news_articles]
        C[model_artifacts]
        D[feature_store]
        E[tick_data]
        F[pattern_instances]
        G[api_logs]
    end

    subgraph "raw_market_data Document"
        A1["_id: ObjectId<br/>symbol: string<br/>timestamp: ISODate<br/>source: string<br/>raw_response: object<br/>metadata: object<br/>created_at: ISODate"]
    end

    subgraph "news_articles Document"
        B1["_id: ObjectId<br/>headline: string<br/>content: text<br/>source: string<br/>published_at: ISODate<br/>symbols: array<br/>sentiment_score: double<br/>entities: array<br/>created_at: ISODate"]
    end

    subgraph "model_artifacts Document"
        C1["_id: ObjectId<br/>model_id: int (ref SQL)<br/>model_name: string<br/>model_type: string<br/>version: string<br/>hyperparameters: object<br/>training_metrics: object<br/>model_binary: Binary/GridFS<br/>created_at: ISODate"]
    end

    subgraph "feature_store Document"
        D1["_id: ObjectId<br/>stock_id: int<br/>timestamp: ISODate<br/>timeframe: string<br/>features: object<br/>- technical: object<br/>- fundamental: object<br/>- sentiment: object<br/>computed_at: ISODate<br/>ttl: ISODate (TTL Index)"]
    end

    A --> A1
    B --> B1
    C --> C1
    D --> D1

    style A fill:#98FB98
    style B fill:#98FB98
    style C fill:#98FB98
    style D fill:#98FB98
```

### Redis Cache Structure

```mermaid
graph TB
    subgraph "Redis Data Structures"
        A[Strings]
        B[Hashes]
        C[Sorted Sets]
        D[Lists]
        E[Streams]
        F[Pub/Sub Channels]
    end

    subgraph "Strings - Simple Values"
        A1["quote:{symbol}<br/>TTL: 5 minutes<br/>Value: JSON quote data"]
        A2["prediction:{symbol}:{timeframe}<br/>TTL: 1 hour<br/>Value: JSON prediction"]
    end

    subgraph "Hashes - Stock Metadata"
        B1["stock:{symbol}<br/>Fields:<br/>- name<br/>- sector<br/>- market_cap<br/>- last_updated"]
    end

    subgraph "Sorted Sets - Leaderboards"
        C1["top_gainers<br/>Score: percent_change<br/>Member: symbol"]
        C2["top_volume<br/>Score: volume<br/>Member: symbol"]
    end

    subgraph "Lists - Recent Alerts"
        D1["alerts:recent<br/>Max Length: 100<br/>Items: JSON alert data"]
    end

    subgraph "Streams - Real-time Events"
        E1["quotes:stream<br/>Events: real-time quotes"]
        E2["alerts:stream<br/>Events: triggered alerts"]
    end

    subgraph "Pub/Sub - Broadcasting"
        F1["quotes:channel<br/>Publishes to UI clients"]
        F2["model:updates<br/>Model version changes"]
    end

    A --> A1
    A --> A2
    B --> B1
    C --> C1
    C --> C2
    D --> D1
    E --> E1
    E --> E2
    F --> F1
    F --> F2

    style A fill:#FFB6C1
    style B fill:#FFB6C1
    style C fill:#FFB6C1
    style D fill:#FFB6C1
    style E fill:#FFB6C1
    style F fill:#FFB6C1
```

---

## Service Communication Patterns

### CQRS Pattern Implementation

```mermaid
graph LR
    subgraph "Commands (Write)"
        A[Create Alert Command]
        B[Update Watchlist Command]
        C[Train Model Command]
    end

    subgraph "Command Handlers"
        D[Alert Command Handler]
        E[Watchlist Command Handler]
        F[Model Training Handler]
    end

    subgraph "Write Database"
        G[(SQL Server<br/>Write Model)]
    end

    subgraph "Event Bus"
        H[Domain Events<br/>MediatR]
    end

    subgraph "Event Handlers"
        I[Cache Invalidation Handler]
        J[Notification Handler]
        K[Analytics Handler]
    end

    subgraph "Queries (Read)"
        L[Get Stock Predictions Query]
        M[Get Alerts Query]
        N[Get Backtest Results Query]
    end

    subgraph "Query Handlers"
        O[Prediction Query Handler]
        P[Alert Query Handler]
        Q[Backtest Query Handler]
    end

    subgraph "Read Database"
        R[(Redis Cache<br/>Read Model)]
        S[(SQL Server<br/>Read Replicas)]
    end

    A --> D
    B --> E
    C --> F

    D --> G
    E --> G
    F --> G

    D --> H
    E --> H
    F --> H

    H --> I
    H --> J
    H --> K

    I --> R

    L --> O
    M --> P
    N --> Q

    O --> R
    O --> S
    P --> R
    P --> S
    Q --> S

    style G fill:#87CEEB
    style R fill:#FFB6C1
    style S fill:#87CEEB
    style H fill:#FFD700
```

---

## Network & Security Architecture

```mermaid
graph TB
    subgraph "External Network"
        A[Internet]
    end

    subgraph "Firewall Layer"
        B[Windows Firewall<br/>Block all external]
        C[Allow localhost only]
    end

    subgraph "Application Ports"
        D[Blazor UI :5000/5001<br/>HTTPS]
        E[Hangfire :5002<br/>Local only]
        F[Seq :5341<br/>Local only]
        G[Grafana :3000<br/>Local only]
    end

    subgraph "Database Ports"
        H[SQL Server :1433<br/>Local only]
        I[MongoDB :27017<br/>Local only]
        J[Redis :6379<br/>Local only]
    end

    subgraph "Outbound Only"
        K[External APIs<br/>HTTPS :443]
    end

    subgraph "Security Measures"
        L[API Key Storage<br/>User Secrets / Env Vars]
        M[HTTPS Only<br/>Self-signed cert]
        N[Connection String Encryption<br/>DPAPI]
        O[SQL Authentication<br/>Windows Auth preferred]
    end

    A --> B
    B --> C
    C --> D
    D --> L
    D --> M

    C --> H
    C --> I
    C --> J

    H --> N
    H --> O

    D --> K

    style B fill:#FF6B6B
    style L fill:#FFD700
    style M fill:#FFD700
    style N fill:#FFD700
    style O fill:#FFD700
```

---

**Document Version**: 1.0
**Last Updated**: 2026-02-03
