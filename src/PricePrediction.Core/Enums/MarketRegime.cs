namespace PricePrediction.Core.Enums;

/// <summary>
/// Market regime states detected by HMM
/// </summary>
public enum MarketRegime
{
    TrendingUp = 1,
    TrendingDown = 2,
    RangeBound = 3,
    HighVolatility = 4,
    LowVolatility = 5
}

/// <summary>
/// Prediction timeframe
/// </summary>
public enum PredictionTimeframe
{
    IntraDay,      // 1-4 hours
    ShortTerm,     // 1-7 days
    MediumTerm,    // 1-4 weeks
    LongTerm       // 1-6 months
}

/// <summary>
/// Pattern type enumeration
/// </summary>
public enum PatternType
{
    // Candlestick Patterns
    Harami,
    BullishEngulfing,
    BearishEngulfing,
    MorningStar,
    EveningStar,
    ThreeWhiteSoldiers,
    ThreeBlackCrows,
    Doji,
    Hammer,
    ShootingStar,

    // Chart Patterns
    HeadAndShoulders,
    InverseHeadAndShoulders,
    DoubleTop,
    DoubleBottom,
    CupAndHandle,
    AscendingTriangle,
    DescendingTriangle,
    SymmetricalTriangle,
    FlagPattern,
    Wedge
}
