using PricePrediction.Core.Models;

namespace PricePrediction.Core.Interfaces;

/// <summary>
/// Feature engineering pipeline interface
/// </summary>
public interface IFeatureEngineer
{
    /// <summary>
    /// Compute all features for a time series
    /// </summary>
    Task<List<FeatureVector>> ComputeFeaturesAsync(
        List<OHLCV> data,
        string symbol,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Compute features for latest data point (online/streaming)
    /// </summary>
    Task<FeatureVector> ComputeFeaturesOnlineAsync(
        OHLCV latest,
        List<OHLCV> historicalContext,
        string symbol,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Update feature importance weights from model feedback
    /// </summary>
    void UpdateFeatureImportance(Dictionary<string, double> importanceScores);
}
