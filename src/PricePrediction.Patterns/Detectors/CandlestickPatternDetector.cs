using PricePrediction.Core.Interfaces;
using PricePrediction.Core.Models;
using PricePrediction.Core.Enums;

namespace PricePrediction.Patterns.Detectors;

/// <summary>
/// Rule-based candlestick pattern detector
/// High-success patterns: Harami (72.85%), Engulfing (58-62%), Morning/Evening Star (60-64%)
/// </summary>
public class CandlestickPatternDetector : IPatternDetector
{
    public string DetectorName => "Candlestick-RuleBased";

    private readonly Dictionary<string, double> _historicalSuccessRates = new()
    {
        { "Harami", 0.7285 },
        { "BullishEngulfing", 0.60 },
        { "BearishEngulfing", 0.60 },
        { "MorningStar", 0.62 },
        { "EveningStar", 0.62 },
        { "ThreeWhiteSoldiers", 0.64 },
        { "ThreeBlackCrows", 0.64 },
        { "Hammer", 0.58 },
        { "ShootingStar", 0.58 },
        { "Doji", 0.52 }
    };

    public async Task<List<PatternDetection>> DetectPatternsAsync(
        List<OHLCV> data,
        string symbol,
        CancellationToken cancellationToken = default)
    {
        return await Task.Run(() =>
        {
            var patterns = new List<PatternDetection>();

            // Need at least 20 bars for volume confirmation
            if (data.Count < 20) return patterns;

            // Calculate average volume for confirmation
            var avgVolume = CalculateAverageVolume(data);

            for (int i = 2; i < data.Count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                // 2-candle patterns
                if (i >= 1)
                {
                    var harami = DetectHarami(data, i, avgVolume);
                    if (harami != null) patterns.Add(harami);

                    var engulfing = DetectEngulfing(data, i, avgVolume);
                    if (engulfing != null) patterns.Add(engulfing);
                }

                // 3-candle patterns
                if (i >= 2)
                {
                    var star = DetectMorningEveningStar(data, i, avgVolume);
                    if (star != null) patterns.Add(star);

                    var soldiers = DetectThreeSoldiersCrows(data, i, avgVolume);
                    if (soldiers != null) patterns.Add(soldiers);
                }

                // Single candle patterns
                var singleCandle = DetectSingleCandlePatterns(data, i, avgVolume);
                patterns.AddRange(singleCandle);
            }

            return patterns;
        }, cancellationToken);
    }

    public async Task<List<PatternDetection>> DetectPatternsOnlineAsync(
        OHLCV latest,
        List<OHLCV> historicalContext,
        string symbol,
        CancellationToken cancellationToken = default)
    {
        var fullData = new List<OHLCV>(historicalContext) { latest };
        var allPatterns = await DetectPatternsAsync(fullData, symbol, cancellationToken);

        // Return only patterns that include the latest candle
        return allPatterns.Where(p => p.EndIndex == fullData.Count - 1).ToList();
    }

    public Task<double> GetPatternSuccessRateAsync(string patternName, string symbol)
    {
        return Task.FromResult(_historicalSuccessRates.GetValueOrDefault(patternName, 0.5));
    }

    #region Pattern Detection Methods

    /// <summary>
    /// Harami pattern - 72.85% success rate
    /// Small candle within previous candle's body
    /// </summary>
    private PatternDetection? DetectHarami(List<OHLCV> data, int index, decimal avgVolume)
    {
        var prev = data[index - 1];
        var current = data[index];

        // Previous candle must have significant body
        if (prev.BodySize < prev.Range * 0.6m) return null;

        // Current candle body must be within previous body
        var prevBodyHigh = System.Math.Max(prev.Open, prev.Close);
        var prevBodyLow = System.Math.Min(prev.Open, prev.Close);
        var currBodyHigh = System.Math.Max(current.Open, current.Close);
        var currBodyLow = System.Math.Min(current.Open, current.Close);

        if (currBodyHigh >= prevBodyHigh || currBodyLow <= prevBodyLow)
            return null;

        // Volume confirmation (+4-6% accuracy)
        var volumeConfirmed = current.Volume > avgVolume * 1.5m;
        var volumeRatio = (double)(current.Volume / avgVolume);

        var isBullish = prev.IsBearish && current.IsBullish;
        var patternName = isBullish ? "BullishHarami" : "BearishHarami";

        return new PatternDetection
        {
            Symbol = data[index].Symbol,
            DetectedAt = current.Timestamp,
            Type = PatternType.Harami,
            Name = patternName,
            StartIndex = index - 1,
            EndIndex = index,
            StartTime = prev.Timestamp,
            EndTime = current.Timestamp,
            DetectionConfidence = volumeConfirmed ? 0.85 : 0.75,
            HistoricalSuccessRate = _historicalSuccessRates["Harami"],
            VolumeConfirmed = volumeConfirmed,
            VolumeRatio = volumeRatio,
            DetectionMethod = "Rule-based",
            ExpectedDirection = isBullish ? 1 : -1,
            Metadata = new()
            {
                { "PrevBodySize", (double)prev.BodySize },
                { "CurrentBodySize", (double)current.BodySize }
            }
        };
    }

    /// <summary>
    /// Engulfing pattern - 58-62% success rate
    /// Current candle's body completely engulfs previous candle's body
    /// </summary>
    private PatternDetection? DetectEngulfing(List<OHLCV> data, int index, decimal avgVolume)
    {
        var prev = data[index - 1];
        var current = data[index];

        // Both candles must have significant bodies
        if (prev.BodySize < prev.Range * 0.5m || current.BodySize < current.Range * 0.6m)
            return null;

        var prevBodyHigh = System.Math.Max(prev.Open, prev.Close);
        var prevBodyLow = System.Math.Min(prev.Open, prev.Close);
        var currBodyHigh = System.Math.Max(current.Open, current.Close);
        var currBodyLow = System.Math.Min(current.Open, current.Close);

        // Current must engulf previous
        if (currBodyHigh <= prevBodyHigh || currBodyLow >= prevBodyLow)
            return null;

        // Opposite colors
        if (prev.IsBullish == current.IsBullish)
            return null;

        var volumeConfirmed = current.Volume > avgVolume * 1.5m;
        var isBullish = current.IsBullish;

        return new PatternDetection
        {
            Symbol = data[index].Symbol,
            DetectedAt = current.Timestamp,
            Type = isBullish ? PatternType.BullishEngulfing : PatternType.BearishEngulfing,
            Name = isBullish ? "BullishEngulfing" : "BearishEngulfing",
            StartIndex = index - 1,
            EndIndex = index,
            StartTime = prev.Timestamp,
            EndTime = current.Timestamp,
            DetectionConfidence = volumeConfirmed ? 0.75 : 0.65,
            HistoricalSuccessRate = isBullish
                ? _historicalSuccessRates["BullishEngulfing"]
                : _historicalSuccessRates["BearishEngulfing"],
            VolumeConfirmed = volumeConfirmed,
            VolumeRatio = (double)(current.Volume / avgVolume),
            DetectionMethod = "Rule-based",
            ExpectedDirection = isBullish ? 1 : -1
        };
    }

    /// <summary>
    /// Morning Star / Evening Star - 60-64% success rate
    /// Three candle reversal pattern
    /// </summary>
    private PatternDetection? DetectMorningEveningStar(List<OHLCV> data, int index, decimal avgVolume)
    {
        var first = data[index - 2];
        var middle = data[index - 1];
        var last = data[index];

        // First and last must have significant bodies
        if (first.BodySize < first.Range * 0.6m || last.BodySize < last.Range * 0.6m)
            return null;

        // Middle must be small (star)
        if (middle.BodySize > middle.Range * 0.3m)
            return null;

        // Check for Morning Star (bullish reversal)
        var isMorningStar = first.IsBearish && last.IsBullish &&
                           middle.High < System.Math.Min(first.Close, last.Open);

        // Check for Evening Star (bearish reversal)
        var isEveningStar = first.IsBullish && last.IsBearish &&
                           middle.Low > System.Math.Max(first.Close, last.Open);

        if (!isMorningStar && !isEveningStar)
            return null;

        var volumeConfirmed = last.Volume > avgVolume * 1.5m;
        var patternName = isMorningStar ? "MorningStar" : "EveningStar";

        return new PatternDetection
        {
            Symbol = data[index].Symbol,
            DetectedAt = last.Timestamp,
            Type = isMorningStar ? PatternType.MorningStar : PatternType.EveningStar,
            Name = patternName,
            StartIndex = index - 2,
            EndIndex = index,
            StartTime = first.Timestamp,
            EndTime = last.Timestamp,
            DetectionConfidence = volumeConfirmed ? 0.80 : 0.70,
            HistoricalSuccessRate = _historicalSuccessRates[patternName],
            VolumeConfirmed = volumeConfirmed,
            VolumeRatio = (double)(last.Volume / avgVolume),
            DetectionMethod = "Rule-based",
            ExpectedDirection = isMorningStar ? 1 : -1
        };
    }

    /// <summary>
    /// Three White Soldiers / Three Black Crows - 62-66% success rate
    /// Three consecutive candles in same direction
    /// </summary>
    private PatternDetection? DetectThreeSoldiersCrows(List<OHLCV> data, int index, decimal avgVolume)
    {
        var first = data[index - 2];
        var second = data[index - 1];
        var third = data[index];

        // All must have significant bodies
        if (first.BodySize < first.Range * 0.6m ||
            second.BodySize < second.Range * 0.6m ||
            third.BodySize < third.Range * 0.6m)
            return null;

        // Three White Soldiers (bullish)
        var isSoldiers = first.IsBullish && second.IsBullish && third.IsBullish &&
                        second.Close > first.Close && third.Close > second.Close &&
                        second.Open > first.Open && second.Open < first.Close &&
                        third.Open > second.Open && third.Open < second.Close;

        // Three Black Crows (bearish)
        var isCrows = first.IsBearish && second.IsBearish && third.IsBearish &&
                     second.Close < first.Close && third.Close < second.Close &&
                     second.Open < first.Open && second.Open > first.Close &&
                     third.Open < second.Open && third.Open > second.Close;

        if (!isSoldiers && !isCrows)
            return null;

        var volumeConfirmed = third.Volume > avgVolume * 1.5m;
        var patternName = isSoldiers ? "ThreeWhiteSoldiers" : "ThreeBlackCrows";

        return new PatternDetection
        {
            Symbol = data[index].Symbol,
            DetectedAt = third.Timestamp,
            Type = isSoldiers ? PatternType.ThreeWhiteSoldiers : PatternType.ThreeBlackCrows,
            Name = patternName,
            StartIndex = index - 2,
            EndIndex = index,
            StartTime = first.Timestamp,
            EndTime = third.Timestamp,
            DetectionConfidence = volumeConfirmed ? 0.78 : 0.68,
            HistoricalSuccessRate = _historicalSuccessRates[patternName],
            VolumeConfirmed = volumeConfirmed,
            VolumeRatio = (double)(third.Volume / avgVolume),
            DetectionMethod = "Rule-based",
            ExpectedDirection = isSoldiers ? 1 : -1
        };
    }

    /// <summary>
    /// Single candle patterns: Hammer, Shooting Star, Doji
    /// </summary>
    private List<PatternDetection> DetectSingleCandlePatterns(List<OHLCV> data, int index, decimal avgVolume)
    {
        var patterns = new List<PatternDetection>();
        var candle = data[index];

        // Doji - indecision
        if (candle.BodySize < candle.Range * 0.1m)
        {
            patterns.Add(new PatternDetection
            {
                Symbol = candle.Symbol,
                DetectedAt = candle.Timestamp,
                Type = PatternType.Doji,
                Name = "Doji",
                StartIndex = index,
                EndIndex = index,
                StartTime = candle.Timestamp,
                EndTime = candle.Timestamp,
                DetectionConfidence = 0.60,
                HistoricalSuccessRate = _historicalSuccessRates["Doji"],
                VolumeConfirmed = candle.Volume > avgVolume * 1.2m,
                VolumeRatio = (double)(candle.Volume / avgVolume),
                DetectionMethod = "Rule-based",
                ExpectedDirection = 0 // Neutral
            });
        }

        // Hammer - bullish reversal
        var lowerWick = System.Math.Min(candle.Open, candle.Close) - candle.Low;
        var upperWick = candle.High - System.Math.Max(candle.Open, candle.Close);

        if (lowerWick > candle.BodySize * 2 && upperWick < candle.BodySize * 0.5m)
        {
            patterns.Add(new PatternDetection
            {
                Symbol = candle.Symbol,
                DetectedAt = candle.Timestamp,
                Type = PatternType.Hammer,
                Name = "Hammer",
                StartIndex = index,
                EndIndex = index,
                StartTime = candle.Timestamp,
                EndTime = candle.Timestamp,
                DetectionConfidence = 0.65,
                HistoricalSuccessRate = _historicalSuccessRates["Hammer"],
                VolumeConfirmed = candle.Volume > avgVolume * 1.3m,
                VolumeRatio = (double)(candle.Volume / avgVolume),
                DetectionMethod = "Rule-based",
                ExpectedDirection = 1
            });
        }

        // Shooting Star - bearish reversal
        if (upperWick > candle.BodySize * 2 && lowerWick < candle.BodySize * 0.5m)
        {
            patterns.Add(new PatternDetection
            {
                Symbol = candle.Symbol,
                DetectedAt = candle.Timestamp,
                Type = PatternType.ShootingStar,
                Name = "ShootingStar",
                StartIndex = index,
                EndIndex = index,
                StartTime = candle.Timestamp,
                EndTime = candle.Timestamp,
                DetectionConfidence = 0.65,
                HistoricalSuccessRate = _historicalSuccessRates["ShootingStar"],
                VolumeConfirmed = candle.Volume > avgVolume * 1.3m,
                VolumeRatio = (double)(candle.Volume / avgVolume),
                DetectionMethod = "Rule-based",
                ExpectedDirection = -1
            });
        }

        return patterns;
    }

    #endregion

    private static decimal CalculateAverageVolume(List<OHLCV> data, int period = 20)
    {
        var takeCount = System.Math.Min(period, data.Count);
        return (decimal)data.TakeLast(takeCount).Average(d => d.Volume);
    }
}
