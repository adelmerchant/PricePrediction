using PricePrediction.Core.Enums;
using PricePrediction.Core.Models;

namespace PricePrediction.Core.Interfaces;

/// <summary>
/// Base interface for all prediction models
/// </summary>
public interface IPredictionModel
{
    string ModelName { get; }
    PredictionTimeframe Timeframe { get; }

    /// <summary>
    /// Train the model on historical data
    /// </summary>
    Task TrainAsync(
        List<FeatureVector> features,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Predict price for a single data point
    /// </summary>
    Task<PredictionResult> PredictAsync(
        FeatureVector features,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Batch prediction for multiple stocks
    /// </summary>
    Task<List<PredictionResult>> PredictBatchAsync(
        List<FeatureVector> features,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Get model accuracy metrics
    /// </summary>
    Task<ModelMetrics> GetMetricsAsync();

    /// <summary>
    /// Save model to disk
    /// </summary>
    Task SaveAsync(string path);

    /// <summary>
    /// Load model from disk
    /// </summary>
    Task LoadAsync(string path);
}

/// <summary>
/// Model performance metrics
/// </summary>
public class ModelMetrics
{
    public string ModelName { get; set; } = string.Empty;
    public double DirectionAccuracy { get; set; }
    public double MAE { get; set; } // Mean Absolute Error
    public double RMSE { get; set; } // Root Mean Squared Error
    public double SharpeRatio { get; set; }
    public double CalibrationError { get; set; } // Expected Calibration Error
    public double Precision { get; set; }
    public double Recall { get; set; }
    public double F1Score { get; set; }
    public DateTime LastUpdated { get; set; }
    public int SampleCount { get; set; }
}
