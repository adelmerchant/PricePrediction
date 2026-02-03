using PricePrediction.Core.Enums;

namespace PricePrediction.Core.Models;

/// <summary>
/// ML model prediction result with confidence intervals
/// </summary>
public class PredictionResult
{
    public string Symbol { get; set; } = string.Empty;
    public DateTime PredictionTime { get; set; }
    public DateTime TargetTime { get; set; }
    public PredictionTimeframe Timeframe { get; set; }

    // Point estimates
    public decimal PredictedPrice { get; set; }
    public decimal CurrentPrice { get; set; }
    public decimal PredictedChange { get; set; }
    public decimal PredictedChangePercent { get; set; }

    // Direction prediction
    public int DirectionPrediction { get; set; } // 1 = Up, 0 = Neutral, -1 = Down
    public double DirectionConfidence { get; set; } // 0-1

    // Confidence intervals (from Monte Carlo Dropout / Quantile Regression)
    public decimal Lower95 { get; set; }
    public decimal Lower80 { get; set; }
    public decimal Upper80 { get; set; }
    public decimal Upper95 { get; set; }

    // Volatility prediction
    public double PredictedVolatility { get; set; }
    public double HistoricalVolatility { get; set; }

    // Model ensemble details
    public Dictionary<string, double> ModelWeights { get; set; } = new();
    public Dictionary<string, double> ModelPredictions { get; set; } = new();

    // Market context
    public MarketRegime CurrentRegime { get; set; }
    public double RegimeConfidence { get; set; }

    // Risk metrics
    public double UncertaintyScore { get; set; } // Higher = more uncertain
    public bool IsHighConfidence => DirectionConfidence > 0.65;
}
