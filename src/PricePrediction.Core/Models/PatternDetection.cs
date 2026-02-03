using PricePrediction.Core.Enums;

namespace PricePrediction.Core.Models;

/// <summary>
/// Detected pattern with metadata
/// </summary>
public class PatternDetection
{
    public string Symbol { get; set; } = string.Empty;
    public DateTime DetectedAt { get; set; }
    public PatternType Type { get; set; }
    public string Name { get; set; } = string.Empty;

    // Pattern location
    public int StartIndex { get; set; }
    public int EndIndex { get; set; }
    public DateTime StartTime { get; set; }
    public DateTime EndTime { get; set; }

    // Confidence metrics
    public double DetectionConfidence { get; set; } // 0-1
    public double HistoricalSuccessRate { get; set; } // 0-1

    // Volume confirmation
    public bool VolumeConfirmed { get; set; }
    public double VolumeRatio { get; set; } // Current volume / Average volume

    // Context
    public bool NearSupport { get; set; }
    public bool NearResistance { get; set; }
    public MarketRegime CurrentRegime { get; set; }

    // Detection method
    public string DetectionMethod { get; set; } = string.Empty; // "Rule-based", "CNN", "DTW"

    // Expected outcome
    public int ExpectedDirection { get; set; } // 1 = Bullish, -1 = Bearish
    public decimal? TargetPrice { get; set; }
    public decimal? StopLoss { get; set; }

    // Pattern-specific data
    public Dictionary<string, object> Metadata { get; set; } = new();
}
