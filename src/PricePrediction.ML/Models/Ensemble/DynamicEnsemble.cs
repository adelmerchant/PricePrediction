using PricePrediction.Core.Enums;
using PricePrediction.Core.Interfaces;
using PricePrediction.Core.Models;
using PricePrediction.Patterns.Regime;

namespace PricePrediction.ML.Models.Ensemble;

/// <summary>
/// Dynamic ensemble orchestrator with regime-aware model weighting
/// Target: 68-74% overall direction accuracy
/// Features:
/// - Dynamic weight calculation based on rolling 20-day accuracy
/// - Regime-aware model switching (HMM)
/// - Minimum weight floor for diversity
/// - Confidence threshold filtering (>0.65)
/// </summary>
public class DynamicEnsemble
{
    private readonly List<IPredictionModel> _models;
    private readonly HmmRegimeDetector _regimeDetector;
    private readonly Dictionary<string, Queue<double>> _rollingAccuracy;
    private readonly Dictionary<string, double> _currentWeights;
    private readonly int _rollingWindow = 20;
    private readonly double _minWeight = 0.05;
    private readonly double _confidenceThreshold = 0.65;
    private readonly double _scalingFactor = 2.0;

    public DynamicEnsemble(List<IPredictionModel> models)
    {
        _models = models ?? throw new ArgumentNullException(nameof(models));
        _regimeDetector = new HmmRegimeDetector(numStates: 3);
        _rollingAccuracy = new Dictionary<string, Queue<double>>();
        _currentWeights = new Dictionary<string, double>();

        // Initialize tracking for each model
        foreach (var model in _models)
        {
            _rollingAccuracy[model.ModelName] = new Queue<double>();
            _currentWeights[model.ModelName] = 1.0 / _models.Count; // Equal weights initially
        }
    }

    /// <summary>
    /// Train HMM regime detector on historical data
    /// </summary>
    public void FitRegimeDetector(List<OHLCV> historicalData)
    {
        if (historicalData.Count < 60)
            throw new ArgumentException("Need at least 60 days of data for regime detection");

        // Calculate features for HMM
        var returns = new double[historicalData.Count - 1];
        var volatilities = new double[historicalData.Count - 1];
        var volumeRatios = new double[historicalData.Count - 1];

        var avgVolume = historicalData.Average(d => (double)d.Volume);

        for (int i = 1; i < historicalData.Count; i++)
        {
            returns[i - 1] = ((double)historicalData[i].Close - (double)historicalData[i - 1].Close) /
                            (double)historicalData[i - 1].Close;

            // Calculate rolling volatility
            var window = historicalData.Skip(System.Math.Max(0, i - 20)).Take(System.Math.Min(i, 20)).ToList();
            var windowReturns = new List<double>();
            for (int j = 1; j < window.Count; j++)
            {
                windowReturns.Add(((double)window[j].Close - (double)window[j - 1].Close) / (double)window[j - 1].Close);
            }
            volatilities[i - 1] = windowReturns.Count > 0 ? StdDev(windowReturns.ToArray()) : 0.01;

            volumeRatios[i - 1] = (double)historicalData[i].Volume / avgVolume;
        }

        _regimeDetector.Fit(returns, volatilities, volumeRatios);
    }

    /// <summary>
    /// Predict with ensemble, using dynamic weights and regime awareness
    /// </summary>
    public async Task<PredictionResult> PredictAsync(
        FeatureVector features,
        OHLCV currentPrice,
        List<OHLCV> recentHistory,
        CancellationToken cancellationToken = default)
    {
        // Get current market regime
        var regime = GetCurrentRegime(recentHistory);

        // Get predictions from all models
        var predictions = new List<PredictionResult>();
        foreach (var model in _models)
        {
            try
            {
                var pred = await model.PredictAsync(features, cancellationToken);
                predictions.Add(pred);
            }
            catch (Exception ex)
            {
                // Log error but continue with other models
                Console.WriteLine($"Model {model.ModelName} prediction failed: {ex.Message}");
            }
        }

        if (predictions.Count == 0)
            throw new InvalidOperationException("All models failed to predict");

        // Update weights based on regime and rolling accuracy
        UpdateWeights(regime, predictions);

        // Combine predictions
        var ensemble = CombinePredictions(predictions, features.Symbol, currentPrice, regime);

        return ensemble;
    }

    /// <summary>
    /// Update model accuracy tracking after observing actual outcome
    /// </summary>
    public void UpdateModelAccuracy(string modelName, bool wasCorrect)
    {
        if (!_rollingAccuracy.ContainsKey(modelName))
            return;

        var queue = _rollingAccuracy[modelName];
        queue.Enqueue(wasCorrect ? 1.0 : 0.0);

        // Maintain rolling window
        while (queue.Count > _rollingWindow)
            queue.Dequeue();
    }

    private void UpdateWeights(MarketRegime regime, List<PredictionResult> predictions)
    {
        var weights = new Dictionary<string, double>();
        var totalWeight = 0.0;

        foreach (var prediction in predictions)
        {
            var modelName = prediction.ModelWeights.Keys.First();

            // Calculate base weight from rolling accuracy
            var accuracy = _rollingAccuracy[modelName].Count > 0
                ? _rollingAccuracy[modelName].Average()
                : 0.5; // Default to 50% if no history

            // Apply regime-based adjustments
            var regimeMultiplier = GetRegimeMultiplier(modelName, regime);
            var weight = System.Math.Exp(accuracy * regimeMultiplier * _scalingFactor);

            weights[modelName] = weight;
            totalWeight += weight;
        }

        // Normalize and apply minimum weight floor
        foreach (var modelName in weights.Keys.ToList())
        {
            var normalizedWeight = weights[modelName] / totalWeight;

            // Apply minimum weight for diversity
            _currentWeights[modelName] = System.Math.Max(normalizedWeight, _minWeight);
        }

        // Re-normalize after applying floor
        var sum = _currentWeights.Values.Sum();
        foreach (var modelName in _currentWeights.Keys.ToList())
        {
            _currentWeights[modelName] /= sum;
        }
    }

    private PredictionResult CombinePredictions(
        List<PredictionResult> predictions,
        string symbol,
        OHLCV currentPrice,
        MarketRegime regime)
    {
        // Weighted voting for direction
        var upVotes = 0.0;
        var downVotes = 0.0;
        var neutralVotes = 0.0;

        var modelPredictions = new Dictionary<string, double>();
        var modelWeights = new Dictionary<string, double>();

        foreach (var pred in predictions)
        {
            var modelName = pred.ModelWeights.Keys.First();
            var weight = _currentWeights.GetValueOrDefault(modelName, 1.0 / predictions.Count);

            modelWeights[modelName] = weight;
            modelPredictions[modelName] = pred.DirectionPrediction;

            if (pred.DirectionPrediction > 0)
                upVotes += weight * pred.DirectionConfidence;
            else if (pred.DirectionPrediction < 0)
                downVotes += weight * pred.DirectionConfidence;
            else
                neutralVotes += weight * pred.DirectionConfidence;
        }

        // Determine ensemble direction
        int ensembleDirection;
        double ensembleConfidence;

        var totalVotes = upVotes + downVotes + neutralVotes;
        if (totalVotes == 0) totalVotes = 1.0;

        if (upVotes > downVotes && upVotes > neutralVotes)
        {
            ensembleDirection = 1;
            ensembleConfidence = upVotes / totalVotes;
        }
        else if (downVotes > upVotes && downVotes > neutralVotes)
        {
            ensembleDirection = -1;
            ensembleConfidence = downVotes / totalVotes;
        }
        else
        {
            ensembleDirection = 0;
            ensembleConfidence = neutralVotes / totalVotes;
        }

        // Weighted price prediction
        var predictedPriceChange = 0.0;
        foreach (var pred in predictions)
        {
            var modelName = pred.ModelWeights.Keys.First();
            var weight = _currentWeights.GetValueOrDefault(modelName, 1.0 / predictions.Count);
            predictedPriceChange += weight * (double)pred.PredictedChangePercent;
        }

        var predictedPrice = (double)currentPrice.Close * (1 + predictedPriceChange / 100);

        // Aggregate confidence intervals (simplified)
        var lower95 = predictions.Min(p => p.Lower95);
        var upper95 = predictions.Max(p => p.Upper95);
        var lower80 = predictions.Min(p => p.Lower80);
        var upper80 = predictions.Max(p => p.Upper80);

        // Calculate uncertainty (wider intervals = more uncertainty)
        var intervalWidth = (double)(upper95 - lower95) / (double)currentPrice.Close;
        var uncertaintyScore = System.Math.Min(intervalWidth * 10, 1.0); // Normalize to 0-1

        return new PredictionResult
        {
            Symbol = symbol,
            PredictionTime = DateTime.UtcNow,
            TargetTime = predictions.First().TargetTime,
            Timeframe = predictions.First().Timeframe,
            CurrentPrice = currentPrice.Close,
            PredictedPrice = (decimal)predictedPrice,
            PredictedChange = (decimal)(predictedPrice - (double)currentPrice.Close),
            PredictedChangePercent = (decimal)predictedPriceChange,
            DirectionPrediction = ensembleDirection,
            DirectionConfidence = ensembleConfidence,
            Lower95 = lower95,
            Lower80 = lower80,
            Upper80 = upper80,
            Upper95 = upper95,
            UncertaintyScore = uncertaintyScore,
            ModelWeights = modelWeights,
            ModelPredictions = modelPredictions,
            CurrentRegime = regime,
            RegimeConfidence = GetRegimeConfidence(predictions)
        };
    }

    private MarketRegime GetCurrentRegime(List<OHLCV> recentHistory)
    {
        if (recentHistory.Count < 2)
            return MarketRegime.RangeBound;

        var latest = recentHistory.Last();
        var previous = recentHistory[^2];

        var currentReturn = ((double)latest.Close - (double)previous.Close) / (double)previous.Close;

        // Calculate recent volatility
        var returns = new List<double>();
        for (int i = 1; i < System.Math.Min(20, recentHistory.Count); i++)
        {
            returns.Add(((double)recentHistory[i].Close - (double)recentHistory[i - 1].Close) /
                       (double)recentHistory[i - 1].Close);
        }
        var volatility = returns.Count > 0 ? StdDev(returns.ToArray()) : 0.01;

        var avgVolume = recentHistory.Average(d => (double)d.Volume);
        var volumeRatio = (double)latest.Volume / avgVolume;

        try
        {
            return _regimeDetector.GetCurrentRegime(currentReturn, volatility, volumeRatio);
        }
        catch
        {
            return MarketRegime.RangeBound;
        }
    }

    private double GetRegimeConfidence(List<PredictionResult> predictions)
    {
        // Calculate agreement between models
        var directions = predictions.Select(p => p.DirectionPrediction).ToList();
        var mostCommon = directions.GroupBy(d => d)
            .OrderByDescending(g => g.Count())
            .First();

        return (double)mostCommon.Count() / directions.Count;
    }

    private double GetRegimeMultiplier(string modelName, MarketRegime regime)
    {
        // Adjust model weights based on regime
        // In production, these would be learned from historical performance
        return regime switch
        {
            MarketRegime.TrendingUp => modelName.Contains("LSTM") || modelName.Contains("GRU") ? 1.2 : 0.9,
            MarketRegime.TrendingDown => modelName.Contains("LSTM") || modelName.Contains("GRU") ? 1.2 : 0.9,
            MarketRegime.RangeBound => modelName.Contains("XGBoost") || modelName.Contains("LightGBM") ? 1.2 : 0.9,
            MarketRegime.HighVolatility => modelName.Contains("GARCH") ? 1.3 : 0.8,
            MarketRegime.LowVolatility => 1.0,
            _ => 1.0
        };
    }

    /// <summary>
    /// Check if ensemble prediction meets confidence threshold
    /// </summary>
    public bool ShouldTrade(PredictionResult prediction)
    {
        return prediction.DirectionConfidence >= _confidenceThreshold &&
               prediction.UncertaintyScore < 0.5;
    }

    private static double StdDev(double[] values)
    {
        if (values.Length == 0) return 0;
        var mean = values.Average();
        var sumSquaredDiffs = values.Select(v => (v - mean) * (v - mean)).Sum();
        return System.Math.Sqrt(sumSquaredDiffs / values.Length);
    }
}
