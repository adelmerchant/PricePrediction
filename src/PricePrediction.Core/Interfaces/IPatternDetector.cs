using PricePrediction.Core.Models;

namespace PricePrediction.Core.Interfaces;

/// <summary>
/// Pattern detection interface
/// </summary>
public interface IPatternDetector
{
    string DetectorName { get; }

    /// <summary>
    /// Detect patterns in historical data
    /// </summary>
    Task<List<PatternDetection>> DetectPatternsAsync(
        List<OHLCV> data,
        string symbol,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Detect patterns in real-time (streaming)
    /// </summary>
    Task<List<PatternDetection>> DetectPatternsOnlineAsync(
        OHLCV latest,
        List<OHLCV> historicalContext,
        string symbol,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Get historical success rate for a pattern
    /// </summary>
    Task<double> GetPatternSuccessRateAsync(
        string patternName,
        string symbol);
}
