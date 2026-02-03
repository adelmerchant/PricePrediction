# Architecture Documentation

This directory contains comprehensive architecture and planning documentation for the US Stock Market Price Prediction Application.

## 📚 Document Index

### 🎯 Start Here

**[EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md)**
- High-level overview of the entire project
- Quick reference for key decisions
- Cost summary and timeline
- **Read this first for a complete picture**

---

### 📋 Detailed Documentation

**[ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md)**
- Detailed system architecture
- Core services description
- Technology stack overview
- Analysis timeframes
- Data sources and providers
- Backtesting strategy
- Azure cloud recommendations

**[SYSTEM_DIAGRAMS.md](SYSTEM_DIAGRAMS.md)**
- High-level architecture diagrams (Mermaid)
- Data flow diagrams
- ML pipeline architecture
- Component interaction diagrams
- Deployment topologies (local, hybrid, cloud)
- Database entity relationships
- Service communication patterns
- Network and security architecture

**[DATABASE_DESIGN.md](DATABASE_DESIGN.md)**
- SQL Server schema design
  - Complete table definitions with indexes
  - Partitioning strategy
  - Columnstore index design
- MongoDB collection schemas
  - Document structure examples
  - Index definitions
  - TTL policies
- Redis cache structure
  - Key naming conventions
  - Data structure design
  - Pub/Sub channels
- Data retention policies

**[TECHNOLOGY_STACK.md](TECHNOLOGY_STACK.md)**
- ML framework comparison (TorchSharp, ML.NET, ONNX)
- Database technology selection
- Frontend framework analysis
- Background job processors
- Technical indicator libraries
- Performance benchmarks
- Recommended package versions
- Alternatives considered and why not chosen

**[COST_ANALYSIS.md](COST_ANALYSIS.md)**
- Phase 1: Local development costs
- Phase 2: Hybrid cloud costs
- Phase 3: Full production costs
- ROI analysis and break-even
- Cost optimization strategies
- Monthly cost calculator

**[IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md)**
- Week-by-week development plan (26 weeks)
- Detailed task breakdown
- Milestones and deliverables
- Risk assessment
- Success criteria
- Testing strategy
- Quick start checklist

---

## 🚀 How to Use These Documents

### For Planning
1. Read **EXECUTIVE_SUMMARY.md** for overall vision
2. Review **COST_ANALYSIS.md** to understand budget
3. Check **IMPLEMENTATION_ROADMAP.md** for timeline

### For Architecture Design
1. Study **ARCHITECTURE_OVERVIEW.md** for system design
2. Reference **SYSTEM_DIAGRAMS.md** for visual architecture
3. Use **DATABASE_DESIGN.md** for schema implementation

### For Technology Selection
1. Read **TECHNOLOGY_STACK.md** for detailed comparisons
2. Check package versions and benchmarks
3. Understand trade-offs of each choice

### For Development
1. Follow **IMPLEMENTATION_ROADMAP.md** week-by-week
2. Reference **DATABASE_DESIGN.md** when creating tables
3. Use **SYSTEM_DIAGRAMS.md** to understand service interactions

---

## 📊 Quick Stats

| Metric | Value |
|--------|-------|
| **Total Documentation** | 6 major documents |
| **Architecture Diagrams** | 15+ diagrams (Mermaid) |
| **Database Tables** | 15+ SQL Server tables, 7+ MongoDB collections |
| **Core Services** | 7 services (Data Fetcher, ML, Trend, Pattern, Prediction, Alerts, Backtesting) |
| **Technology Stack** | .NET 10, TorchSharp, SQL Server, MongoDB, Redis |
| **Development Timeline** | 26 weeks (part-time) or 13 weeks (full-time) |
| **Phase 1 Cost** | $163/month (data + electricity) |
| **Phase 2 Cost** | $599/month (hybrid cloud, no news) |

---

## 🎯 Key Architectural Decisions

### Technology Choices

| Component | Choice | Why |
|-----------|--------|-----|
| **Language** | C# (.NET 10) | Unified stack, GPU support, modern |
| **ML Training** | TorchSharp + CUDA | Full PyTorch in .NET, GPU acceleration |
| **ML Inference** | ONNX Runtime GPU | Fastest inference (2-5x faster) |
| **Frontend** | Blazor Server → WASM | .NET integration, real-time updates |
| **Database (Relational)** | SQL Server 2022 | Columnstore indexes, free Dev edition |
| **Database (NoSQL)** | MongoDB Community | Time-series, versatile, free |
| **Cache** | Redis 7.x | Pub/Sub, rich structures, persistence |
| **Data Provider** | Polygon.io | Best value, comprehensive data |

### Deployment Strategy

| Phase | Deployment | Timeline | Cost/Month |
|-------|------------|----------|------------|
| **Phase 1** | 100% Local Workstation | Months 0-6 | $163 |
| **Phase 2** | Hybrid (UI cloud, ML local) | Months 6-18 | $599 |
| **Phase 3** | Full Cloud (if SaaS) | Months 18+ | $2,606+ |

---

## 🏗️ System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Blazor Server UI                         │
│              (MudBlazor + TradingView Charts)               │
└─────────────────────────────────────────────────────────────┘
                           ↓ SignalR + REST API
┌─────────────────────────────────────────────────────────────┐
│                   Application Services                       │
│  ┌─────────────┬──────────────┬─────────────┬────────────┐ │
│  │ Data Fetcher│ Trend Detect │ Pattern Det │ Prediction │ │
│  ├─────────────┼──────────────┼─────────────┼────────────┤ │
│  │ Alert Proc  │ Backtesting  │             │            │ │
│  └─────────────┴──────────────┴─────────────┴────────────┘ │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│               ML Pipeline (GPU-Accelerated)                  │
│  ┌──────────────┬────────────────┬──────────────────────┐  │
│  │ TorchSharp   │ ML.NET         │ ONNX Runtime         │  │
│  │ (Training)   │ (XGBoost)      │ (Inference)          │  │
│  │ LSTM, GRU    │ LightGBM       │ GPU-optimized        │  │
│  └──────────────┴────────────────┴──────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                      Data Storage                            │
│  ┌─────────────┬────────────────┬──────────────────────┐   │
│  │ SQL Server  │ MongoDB        │ Redis                │   │
│  │ (OHLCV)     │ (Raw data)     │ (Cache)              │   │
│  └─────────────┴────────────────┴──────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 ML Model Portfolio

| Timeframe | Model | Architecture | Target Accuracy |
|-----------|-------|--------------|-----------------|
| **Intra-day** (5min-1hr) | LSTM + Attention | 3 layers, 128 hidden | >60% directional |
| **Short-term** (1-30 days) | GRU + XGBoost | Ensemble | >65% directional |
| **Long-term** (1-12 months) | Prophet + Fundamentals | Trend decomposition | Sharpe >1.5 |
| **Pattern Detection** | CNN | 5 conv layers | >70% precision |

---

## 📅 Development Phases

### Phase 1: Foundation (Weeks 1-6)
- ✅ Environment setup
- ✅ Database design
- ✅ Data fetcher service
- ✅ Background job scheduling
- ✅ Historical data loading

### Phase 2: ML Pipeline (Weeks 7-12)
- Feature engineering
- Model training (LSTM, GRU, XGBoost)
- ONNX export and inference
- Model monitoring

### Phase 3: Services (Weeks 13-18)
- Trend detection
- Pattern detection
- Price prediction
- Alert processor
- Backtesting engine

### Phase 4: UI (Weeks 19-22)
- Blazor UI with MudBlazor
- Dashboard and charts
- Predictions and alerts UI
- Backtesting interface

### Phase 5: Testing (Weeks 23-26)
- Unit tests (70%+ coverage)
- Integration tests
- Performance optimization
- Security audit

### Phase 6: Deployment (Week 27+)
- Production deployment
- Monitoring and logging
- Continuous improvement

---

## 💡 Best Practices

### When Reading These Docs

1. **Start with Executive Summary** - Get the big picture
2. **Dive into specifics** as needed for implementation
3. **Reference diagrams** when building components
4. **Check cost analysis** before adding cloud services
5. **Follow roadmap** to stay on track

### When Implementing

1. **Stick to the architecture** - Don't reinvent the wheel
2. **Use recommended technologies** - They've been vetted
3. **Follow database schemas** - Consistency is key
4. **Test early and often** - Per roadmap testing strategy
5. **Document deviations** - If you change something, note why

### When Estimating

1. **Use cost calculator** - Don't guess at cloud costs
2. **Add 20% buffer** - Things always cost more than expected
3. **Start small** - Phase 1 before Phase 2
4. **Monitor actual costs** - Optimize based on reality

---

## 🔗 External References

### Data Providers
- [Polygon.io](https://polygon.io) - Stock market data API
- [Financial Modeling Prep](https://financialmodelingprep.com) - Fundamental data
- [Alpha Vantage](https://www.alphavantage.co) - Backup data source

### ML Frameworks
- [TorchSharp](https://github.com/dotnet/TorchSharp) - PyTorch for .NET
- [ML.NET](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet) - Microsoft ML framework
- [ONNX Runtime](https://onnxruntime.ai) - Optimized inference

### Libraries & Tools
- [TA-Lib](https://ta-lib.org) - Technical analysis library
- [MudBlazor](https://mudblazor.com) - Blazor component library
- [TradingView](https://www.tradingview.com/HTML5-stock-forex-bitcoin-charting-library/) - Lightweight charts
- [Quartz.NET](https://www.quartz-scheduler.net) - Job scheduling
- [Hangfire](https://www.hangfire.io) - Background processing

---

## 📝 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-03 | Initial architecture documentation |

---

## 🤝 Contributing

This is a planning document for a single-user application. If you're adapting this architecture for your own project:

1. Fork the repository
2. Modify architecture to fit your needs
3. Document your changes
4. Share your learnings (optional)

---

## 📞 Questions & Support

For questions about this architecture:
- Review the **EXECUTIVE_SUMMARY.md** first
- Check the specific document for your topic
- Consult the diagrams in **SYSTEM_DIAGRAMS.md**

---

## 🎯 Next Steps

**Ready to start building?**

1. ✅ Read **EXECUTIVE_SUMMARY.md** (done reading this? Good!)
2. ⏭️ Go to **IMPLEMENTATION_ROADMAP.md**
3. 🚀 Follow Week 1 tasks (Environment Setup)
4. 💻 Start coding!

**Need more details?**
- Architecture: See **ARCHITECTURE_OVERVIEW.md**
- Database: See **DATABASE_DESIGN.md**
- Technology: See **TECHNOLOGY_STACK.md**
- Costs: See **COST_ANALYSIS.md**

---

**Happy Building! 🚀**

---

*Last Updated: February 3, 2026*
*Status: Architecture Complete - Ready for Development*
