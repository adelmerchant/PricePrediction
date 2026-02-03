# Executive Summary - US Stock Market Price Prediction Application

## Overview

This document provides a high-level summary of the complete architecture and plan for building a GPU-accelerated stock market price prediction application for single-user deployment on a local workstation, with a future path to cloud-based multi-user SaaS.

**Document Date**: February 3, 2026
**Architecture Version**: 1.0
**Status**: Planning Complete - Ready for Development

---

## Project Vision

Build a comprehensive stock market analysis and prediction platform that leverages machine learning, technical analysis, and pattern recognition to provide actionable trading insights across multiple timeframes (intra-day, short-term, long-term).

### Key Differentiators
- **GPU-Accelerated ML**: Utilize NVIDIA RTX 3090 for fast model training and inference
- **Multi-Timeframe Analysis**: Cover intra-day scalping to long-term investing
- **Unified Technology Stack**: Pure .NET 10 from backend to frontend
- **Cost-Effective**: Start local, scale to cloud only when needed
- **Extensible**: Architecture supports future expansion to cryptocurrency

---

## System Architecture at a Glance

### Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Frontend** | Blazor Server (.NET 10) | Interactive web UI with real-time updates |
| **Backend** | ASP.NET Core 10 | REST APIs, SignalR, business logic |
| **ML Training** | TorchSharp + CUDA 12.x | Deep learning (LSTM, GRU, Transformers) |
| **ML Inference** | ONNX Runtime GPU | Optimized model serving |
| **Traditional ML** | ML.NET + LightGBM | Tree-based models with GPU acceleration |
| **Relational DB** | SQL Server 2022 Developer | OHLCV data, predictions, backtests |
| **Document DB** | MongoDB 6.0+ Community | Raw data, news, model artifacts |
| **Cache** | Redis 7.x | Real-time quotes, predictions cache |
| **Background Jobs** | Quartz.NET + Hangfire | Data fetching, model training, alerts |
| **Technical Analysis** | TA-Lib.NETCore | 150+ indicators |
| **Charting** | TradingView Lightweight Charts | Professional financial charts |

---

### Core Services

```
┌─────────────────────────────────────────────────────────┐
│  UI Layer: Blazor Server + MudBlazor + TradingView     │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  Application Services:                                  │
│  • Data Fetcher (Polygon.io, Alpha Vantage)            │
│  • Trend Detection (Multi-timeframe analysis)          │
│  • Pattern Detection (Chart + Candlestick patterns)    │
│  • Price Prediction (Ensemble of ML models)            │
│  • Alert Processor (Rule engine + notifications)       │
│  • Backtesting Engine (Walk-forward validation)        │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  ML Pipeline: TorchSharp → Training → ONNX → Inference │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  Data Layer: SQL Server + MongoDB + Redis              │
└─────────────────────────────────────────────────────────┘
```

---

## Hardware Requirements

### Your Current Setup (RTX 3090)
✅ **Excellent for this project!**

The NVIDIA RTX 3090 with 24 GB VRAM is perfect for:
- Training large LSTM/GRU models
- Batch inference across hundreds of stocks
- Running multiple model variations simultaneously

### Recommended Workstation Specs

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **GPU** | RTX 3060 Ti (8 GB) | **RTX 3090 (24 GB)** ✅ | RTX 4090 (24 GB) |
| **CPU** | 8-core 3.0+ GHz | 16-core 3.5+ GHz | 16-core 4.0+ GHz |
| **RAM** | 32 GB DDR4 | **64 GB DDR4** ✅ | 128 GB DDR5 |
| **Storage (OS)** | 500 GB NVMe | 1 TB NVMe Gen3 | 2 TB NVMe Gen4 |
| **Storage (Data)** | 1 TB NVMe | **2 TB NVMe Gen3** ✅ | 4 TB NVMe Gen4 |
| **PSU** | 750W 80+ | 850W 80+ Gold | 1000W 80+ Gold |
| **UPS** | - | 1500VA / 900W | 1500VA / 900W |

**Your Status**: Already have RTX 3090 (saves $1,100!). Ensure you have:
- ✅ 64 GB RAM (32 GB minimum)
- ✅ 2-3 TB fast NVMe storage
- ✅ 850W+ PSU
- 🔲 UPS (recommended: $175)

---

## Cost Analysis

### Phase 1: Local Development (Months 0-6)

| Category | Cost |
|----------|------|
| **Hardware** (if needed) | $0 - $1,095 |
| **Software** | $0 (all free/open-source) |
| **Data Providers** | $113/month ($1,356/year) |
| **Electricity** | $50/month ($600/year) |
| **Total Year 1** | **$1,956 - $3,190** |

**Best Case**: $163/month if no hardware upgrades needed

---

### Phase 2: Hybrid Cloud (Months 6-18)

| Category | Cost/Month | Cost/Year |
|----------|------------|-----------|
| **Azure Services** | $281 | $3,372 |
| **Data Providers** (no news) | $268 | $3,216 |
| **Data Providers** (with news) | $717 | $8,604 |
| **Local Workstation** | $50 | $600 |
| **Total (no news)** | **$599** | **$7,188** |
| **Total (with news)** | **$1,048** | **$12,576** |

**Recommended**: Start without news service ($599/month)

---

### Phase 3: Multi-User SaaS (Future)

| Metric | Value |
|--------|-------|
| **Operating Costs** | $2,606 - $3,919/month |
| **Target Users** | 100 active users |
| **Subscription Price** | $49 - $69/user/month |
| **Monthly Revenue** | $4,900 - $6,900 |
| **Monthly Profit** | $1,000 - $4,300 |
| **Profit Margin** | 30% - 47% |

---

## Data Sources

### Recommended Providers

#### Phase 1 (Development)
- **Primary**: Polygon.io Stocks Starter ($99/month)
  - Real-time quotes, historical data, 1-min bars
  - 5 API calls/sec
- **Fundamental**: Financial Modeling Prep Professional ($14/month)
  - Financial statements, ratios
- **Backup**: Yahoo Finance (free via yfinance.NET)

**Total**: $113/month

#### Phase 2 (Production)
- **Primary**: Polygon.io Stocks Advanced ($199/month)
  - 100 API calls/sec
- **Fundamental**: Financial Modeling Prep Enterprise ($69/month)
- **News** (optional): NewsAPI.org Business ($449/month)

**Total**: $268/month (no news) or $717/month (with news)

---

## Machine Learning Models

### Model Portfolio

| Timeframe | Model Type | Architecture | Use Case |
|-----------|------------|--------------|----------|
| **Intra-day** (5min-1hr) | LSTM + Attention | 3 layers, 128 hidden, 60 seq | Day trading, scalping |
| **Short-term** (1-30 days) | GRU + XGBoost Ensemble | GRU: 2 layers, 64 hidden<br/>XGBoost: 1000 trees | Swing trading, weekly options |
| **Long-term** (1-12 months) | Prophet + Fundamentals | Trend decomposition + ratios | Portfolio allocation, investing |
| **Pattern Detection** | CNN | 5 conv layers | Chart pattern recognition |

### Expected Performance

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| **Directional Accuracy** | >60% | >65% |
| **Sharpe Ratio** (backtests) | >1.0 | >1.5 |
| **Max Drawdown** | <20% | <15% |
| **Inference Latency** | <50ms | <30ms |

**Note**: Stock market prediction is inherently challenging. Focus on risk-adjusted returns, not absolute accuracy.

---

## Analysis Capabilities

### Timeframe Coverage

1. **Intra-Day Analysis** (Minutes to Hours)
   - 1-min, 5-min, 15-min, 1-hour bars
   - Real-time quote updates every 1-5 minutes
   - Technical indicators: Fast RSI, VWAP, volume profile
   - Use Case: Day trading, scalping

2. **Short-Term Analysis** (1-30 Days)
   - Hourly, 4-hour, daily bars
   - Update frequency: Hourly to daily
   - Technical indicators: MACD, Bollinger Bands, Stochastic
   - Chart patterns: Flags, triangles, head & shoulders
   - Use Case: Swing trading, weekly options

3. **Long-Term Analysis** (1-12 Months)
   - Daily, weekly bars
   - Update frequency: Daily to weekly
   - Fundamental analysis: P/E, P/B, ROE, debt ratios
   - Macro indicators: GDP, inflation, interest rates
   - Use Case: Investment portfolio, retirement accounts

---

## Development Timeline

### Realistic Schedule (Part-Time Development)

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| **Phase 1**: Foundation & Data | Weeks 1-6 | Data pipeline, database, background jobs |
| **Phase 2**: ML Pipeline | Weeks 7-12 | Models trained, inference service |
| **Phase 3**: Core Services | Weeks 13-18 | Trend, pattern, prediction, alerts |
| **Phase 4**: User Interface | Weeks 19-22 | Blazor UI, charts, dashboards |
| **Phase 5**: Testing & QA | Weeks 23-26 | Unit tests, performance tuning |
| **Phase 6**: Deployment | Week 27-28 | Production deployment |

**Total Duration**: ~6-7 months (part-time) or ~3-4 months (full-time)

---

### Critical Milestones

- ✅ **M0**: Architecture Planning Complete (Week 0)
- 🔲 **M1**: Data Pipeline Operational (Week 6)
- 🔲 **M2**: First ML Model Trained (Week 10)
- 🔲 **M3**: All Services Complete (Week 18)
- 🔲 **M4**: UI Complete (Week 22)
- 🔲 **M5**: Production Ready (Week 26)
- 🔲 **M6**: Live in Production (Week 28)

---

## Key Features

### MVP (Minimum Viable Product)
1. ✅ Real-time quote fetching for 100+ stocks
2. ✅ Historical data storage (1+ year daily, 3+ months intraday)
3. ✅ Intraday LSTM prediction model
4. ✅ Short-term ensemble model (GRU + XGBoost)
5. ✅ Trend detection (multi-timeframe)
6. ✅ Basic pattern detection (10+ patterns)
7. ✅ Alert system (price targets, pattern detection)
8. ✅ Web UI with charts and dashboards
9. ✅ Backtesting engine

### Post-MVP Enhancements
- Long-term prediction model with fundamentals
- News sentiment analysis integration
- Advanced chart pattern detection (20+ patterns)
- Portfolio optimization
- Options strategy recommendations
- Mobile app (Blazor MAUI or React Native)
- Bitcoin and cryptocurrency support
- Multi-user authentication (Azure AD B2C)
- Social features (share watchlists, strategies)

---

## Risk Mitigation

### Top Risks & Mitigations

1. **Model Accuracy Insufficient**
   - Mitigation: Use ensemble methods, focus on directional accuracy, set realistic expectations

2. **API Rate Limits**
   - Mitigation: Aggressive caching, multiple providers, fetch only watchlist stocks initially

3. **GPU Memory Overflow**
   - Mitigation: Monitor VRAM, reduce batch size, use gradient checkpointing, RTX 3090 has 24 GB (ample)

4. **Scope Creep**
   - Mitigation: Stick to roadmap, defer non-critical features, MVP-first approach

5. **Data Quality Issues**
   - Mitigation: Validation pipeline, cross-check sources, anomaly detection

---

## Success Criteria

### Phase 1 Success (Local Development)
- [ ] Can fetch and store real-time data for 100 stocks
- [ ] 1 year of daily data + 3 months of intraday data loaded
- [ ] At least one ML model achieves >55% directional accuracy
- [ ] Backtesting framework produces valid results
- [ ] UI displays charts and predictions

### Phase 2 Success (Hybrid Cloud)
- [ ] UI accessible remotely via Azure App Service
- [ ] System handles 10 concurrent users (if shared)
- [ ] 99.5% uptime over 30 days
- [ ] Model predictions cached and served in <100ms
- [ ] Total operating cost under $650/month

### Long-Term Success (Business Goals)
- [ ] Consistently profitable trading signals (>1.5 Sharpe ratio)
- [ ] 50+ active users (if SaaS)
- [ ] Positive profit margin (>30%)
- [ ] Automated model retraining working reliably
- [ ] Bitcoin integration complete

---

## Deployment Strategy

### Phase 1: Local Workstation (Now - Month 6)
**Architecture**: 100% local
- All services on Windows 11 workstation
- SQL Server, MongoDB, Redis on same machine
- Blazor Server UI accessed via localhost:5000
- Background jobs (Quartz.NET, Hangfire) running locally

**Pros**: Zero cloud costs, low latency, full control
**Cons**: Not accessible remotely, single point of failure

---

### Phase 2: Hybrid Cloud (Month 6-18)
**Architecture**: UI in cloud, ML on local
- Blazor WebAssembly UI on Azure App Service
- REST APIs on Azure Container Apps
- Databases (SQL, MongoDB, Redis) on Azure
- **ML training and inference still on local GPU workstation**
- Model artifacts synced to Azure Blob Storage

**Pros**: Remote access, scalable UI, cost-effective ML
**Cons**: Network latency to local ML, complexity

---

### Phase 3: Full Cloud (Month 18+, if SaaS)
**Architecture**: Everything in Azure
- App Service for UI
- Azure SQL Database (Business Critical for HA)
- Cosmos DB (MongoDB API)
- Azure Cache for Redis (Premium)
- Azure ML or GPU VMs for model training

**Pros**: Full scalability, high availability, no local dependencies
**Cons**: Expensive (~$2,600-3,900/month), GPU compute costs

---

## Backtesting Approach

### Walk-Forward Validation

```
Training Window: 252 days (1 year)
Validation Window: 63 days (3 months)
Test Window: 21 days (1 month)

[-----Train (252d)-----][--Val (63d)--][Test (21d)]
      ↓ Roll forward
      [-----Train (252d)-----][--Val (63d)--][Test (21d)]
            ↓ Roll forward
            [-----Train (252d)-----][--Val (63d)--][Test (21d)]
```

### Performance Metrics
- **Returns**: Total, annualized, monthly
- **Risk**: Sharpe, Sortino, Max Drawdown, VaR
- **Trades**: Win rate, profit factor, avg win/loss
- **Model**: Accuracy, MAE, RMSE, directional accuracy

### Transaction Costs
- Commission: $0 per trade (most modern brokers)
- Slippage: 0.05% for liquid stocks, 0.1-0.5% for illiquid
- Spread: Bid-ask spread from historical data

---

## Next Steps

### Immediate Actions (Week 1)

1. **Environment Setup** (Day 1-2)
   - [ ] Install Visual Studio 2022 Community
   - [ ] Install SQL Server 2022 Developer Edition
   - [ ] Install MongoDB Community 6.0+
   - [ ] Install Redis 7.x (Windows port or Docker)
   - [ ] Install CUDA Toolkit 12.x + cuDNN 8.x
   - [ ] Verify GPU: `nvidia-smi` shows RTX 3090

2. **Project Initialization** (Day 3)
   - [ ] Create GitHub repository (or use existing)
   - [ ] Create .NET 10 solution structure
   - [ ] Add projects: Web, Services, Data, ML, Tests
   - [ ] Setup .gitignore for .NET, user secrets
   - [ ] Initial commit

3. **Data Provider Setup** (Day 4)
   - [ ] Sign up for Polygon.io account (Starter plan or free trial)
   - [ ] Sign up for Financial Modeling Prep (free tier to start)
   - [ ] Test API access (fetch a single quote)
   - [ ] Store API keys in User Secrets

4. **Database Setup** (Day 5)
   - [ ] Create SQL Server database (PricePrediction)
   - [ ] Create MongoDB database (price_prediction)
   - [ ] Test connections from .NET
   - [ ] Initialize schema (first migrations)

5. **First Code** (Day 6-7)
   - [ ] Implement simple Polygon.io client
   - [ ] Fetch quote for AAPL and store in SQL Server
   - [ ] Verify data in database
   - [ ] Setup logging (Serilog)

**Week 1 Success**: Can fetch and store a single stock quote from API to database

---

### Weekly Goals (Weeks 2-6)

- **Week 2**: Data fetcher service complete (all symbols)
- **Week 3**: Historical data loading (500 stocks, 1 year)
- **Week 4**: Background jobs (scheduled data fetching)
- **Week 5**: Feature engineering pipeline (technical indicators)
- **Week 6**: First LSTM model trained

---

## Documentation Index

All architecture documentation is located in `docs/architecture/`:

1. **ARCHITECTURE_OVERVIEW.md** (This summary)
   - High-level system design
   - Core services overview
   - Technology stack

2. **SYSTEM_DIAGRAMS.md**
   - Architecture diagrams (Mermaid)
   - Data flow diagrams
   - Component interaction
   - Deployment topology

3. **DATABASE_DESIGN.md**
   - SQL Server schema (tables, indexes, partitions)
   - MongoDB collections
   - Redis cache structure
   - Data retention policies

4. **TECHNOLOGY_STACK.md**
   - Detailed technology comparisons
   - Framework justifications
   - Performance benchmarks
   - Package versions

5. **COST_ANALYSIS.md**
   - Phase-by-phase cost breakdown
   - ROI analysis
   - Cost optimization strategies
   - Budget planning

6. **IMPLEMENTATION_ROADMAP.md**
   - Week-by-week task breakdown
   - Milestones and deliverables
   - Risk assessment
   - Testing strategy

---

## Key Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Primary Language** | C# (.NET 10) | Unified stack, GPU support, modern language |
| **Frontend** | Blazor Server (Phase 1) → Blazor WASM (Phase 2) | .NET integration, real-time via SignalR |
| **ML Training** | TorchSharp + CUDA | Full PyTorch features, GPU acceleration |
| **ML Inference** | ONNX Runtime GPU | Fastest inference, cross-framework |
| **Relational DB** | SQL Server 2022 | Columnstore indexes, Windows native, free Dev edition |
| **Document DB** | MongoDB Community | Time-series collections, free, versatile |
| **Cache** | Redis 7.x | Pub/Sub, rich data structures, persistence |
| **Data Provider** | Polygon.io (primary) | Best value, real-time + historical |
| **Charting** | TradingView Lightweight Charts | Professional, performant, finance-specific |
| **UI Components** | MudBlazor | Material Design, rich components, active |
| **Background Jobs** | Quartz.NET + Hangfire | Scheduling + dashboard + ad-hoc jobs |
| **Deployment (Phase 1)** | Local workstation | Zero cloud costs, leverage RTX 3090 |
| **Deployment (Phase 2)** | Hybrid (UI cloud, ML local) | Remote access, cost-effective ML |

---

## Contact & Support

**Project Type**: Personal/Solo Development
**Repository**: https://github.com/[your-username]/PricePrediction
**Documentation**: `/docs/architecture/`
**Issues/Bugs**: GitHub Issues

---

## Appendix: Quick Reference

### Useful Commands

**Check GPU**:
```bash
nvidia-smi
```

**Run SQL Server**:
```bash
# Already running as Windows service
```

**Run MongoDB**:
```bash
mongod --dbpath C:\data\db
```

**Run Redis**:
```bash
redis-server
```

**Run Application**:
```bash
cd src/PricePrediction.Web
dotnet run
```

**Run Hangfire Dashboard**:
Navigate to: https://localhost:5002/hangfire

**View Logs (Seq)**:
Navigate to: http://localhost:5341

---

### Important URLs (Local Development)

- **Blazor UI**: https://localhost:5000
- **REST API**: https://localhost:5001/api
- **Swagger**: https://localhost:5001/swagger
- **Hangfire Dashboard**: https://localhost:5002/hangfire
- **Seq Logs**: http://localhost:5341

---

### Database Connection Strings (Template)

**SQL Server**:
```
Server=localhost;Database=PricePrediction;Integrated Security=true;TrustServerCertificate=true;
```

**MongoDB**:
```
mongodb://localhost:27017/price_prediction
```

**Redis**:
```
localhost:6379
```

---

## Final Thoughts

This architecture provides a **solid, scalable foundation** for a stock market prediction application. Key strengths:

1. ✅ **Cost-Effective**: Start with $163/month (data + electricity)
2. ✅ **GPU-Optimized**: Leverage your RTX 3090 to the fullest
3. ✅ **Modern Stack**: .NET 10, latest ML frameworks
4. ✅ **Scalable**: Clear path from local to cloud to SaaS
5. ✅ **Extensible**: Easy to add Bitcoin, options, forex later
6. ✅ **Well-Documented**: Comprehensive architecture docs

The roadmap is ambitious but achievable with disciplined execution. **Focus on MVP first**—get a working system with one good model, then iterate.

### Recommended Approach

1. **Weeks 1-12**: Build MVP (data + one model + basic UI)
2. **Weeks 13-18**: Add services (alerts, backtesting)
3. **Weeks 19-22**: Polish UI
4. **Weeks 23-26**: Test and optimize
5. **Week 27+**: Deploy and use daily

**Remember**: Stock prediction is hard. Manage expectations. Focus on:
- Risk-adjusted returns (Sharpe ratio >1.5)
- Directional accuracy (>60%) not exact prices
- Portfolio optimization, not single-stock prediction
- Learning and iteration

---

**Good luck with development!**

---

**Document Version**: 1.0
**Status**: ✅ Architecture Complete - Ready to Build
**Next Review**: After Phase 1 completion (Week 6)
