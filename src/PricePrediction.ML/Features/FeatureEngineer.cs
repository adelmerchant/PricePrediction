using PricePrediction.Core.Interfaces;
using PricePrediction.Core.Models;
using PricePrediction.ML.Indicators;
using PricePrediction.Math.Filters;
using PricePrediction.Math.Statistics;
using PricePrediction.Math.Volatility;

namespace PricePrediction.ML.Features;

/// <summary>
/// Core feature engineering pipeline
/// Generates 53 features from OHLCV data
/// </summary>
public class FeatureEngineer : IFeatureEngineer
{
    private readonly KalmanFilter _kalmanFilter;
    private readonly GarchModel _garchModel;
    private Dictionary<string, double> _featureImportance = new();

    public FeatureEngineer()
    {
        _kalmanFilter = new KalmanFilter();
        _garchModel = new GarchModel();
    }

    public async Task<List<FeatureVector>> ComputeFeaturesAsync(
        List<OHLCV> data,
        string symbol,
        CancellationToken cancellationToken = default)
    {
        if (data.Count < 250) // Need at least ~1 year of data
            throw new ArgumentException("Insufficient data for feature computation. Need at least 250 data points.");

        return await Task.Run(() =>
        {
            var features = new List<FeatureVector>();

            // Extract arrays
            var closes = data.Select(d => (double)d.Close).ToArray();
            var opens = data.Select(d => (double)d.Open).ToArray();
            var highs = data.Select(d => (double)d.High).ToArray();
            var lows = data.Select(d => (double)d.Low).ToArray();
            var volumes = data.Select(d => (double)d.Volume).ToArray();

            // === Compute all indicators ===

            // Momentum
            var rsi7 = TechnicalIndicators.RSI(closes, 7);
            var rsi14 = TechnicalIndicators.RSI(closes, 14);
            var rsi21 = TechnicalIndicators.RSI(closes, 21);
            var (macd, macdSignal, macdHist) = TechnicalIndicators.MACD(closes);
            var roc5 = TechnicalIndicators.ROC(closes, 5);
            var roc10 = TechnicalIndicators.ROC(closes, 10);
            var roc20 = TechnicalIndicators.ROC(closes, 20);
            var (stochK, stochD) = TechnicalIndicators.Stochastic(highs, lows, closes);
            var williamsR = TechnicalIndicators.WilliamsR(highs, lows, closes);

            // Trend
            var sma10 = TechnicalIndicators.SMA(closes, 10);
            var sma20 = TechnicalIndicators.SMA(closes, 20);
            var sma50 = TechnicalIndicators.SMA(closes, 50);
            var sma100 = TechnicalIndicators.SMA(closes, 100);
            var sma200 = TechnicalIndicators.SMA(closes, 200);
            var ema12 = TechnicalIndicators.EMA(closes, 12);
            var ema26 = TechnicalIndicators.EMA(closes, 26);
            var ema9 = TechnicalIndicators.EMA(closes, 9);
            var ema21 = TechnicalIndicators.EMA(closes, 21);
            var (adx, plusDI, minusDI) = TechnicalIndicators.ADX(highs, lows, closes);
            var linearSlope = TechnicalIndicators.LinearRegressionSlope(closes, 20);

            // Volatility
            var atr14 = TechnicalIndicators.ATR(highs, lows, closes, 14);
            var atr20 = TechnicalIndicators.ATR(highs, lows, closes, 20);
            var (bbUpper, bbMiddle, bbLower, bbWidth, bbPercent) = TechnicalIndicators.BollingerBands(closes);
            var parkinsonVol = TechnicalIndicators.ParkinsonVolatility(highs, lows, 20);

            // Volume
            var obv = TechnicalIndicators.OBV(closes, volumes);
            var vwap = TechnicalIndicators.VWAP(highs, lows, closes, volumes);
            var ad = TechnicalIndicators.AccumulationDistribution(highs, lows, closes, volumes);

            // Mathematical models
            var kalmanResults = _kalmanFilter.FilterSeries(closes);
            var returns = GarchModel.CalculateReturns(closes);

            double[] garchVol = new double[closes.Length];
            if (returns.Length >= 60)
            {
                try
                {
                    garchVol = _garchModel.FitTransform(returns);
                }
                catch
                {
                    // If GARCH fails, use rolling std
                    garchVol = RollingStd(returns, 20);
                }
            }

            var hurstValues = closes.Length >= 120
                ? HurstExponent.CalculateRolling(closes, 60)
                : Enumerable.Repeat(0.5, closes.Length).ToArray();

            // === Build feature vectors ===
            var startIdx = 250; // Skip initial period where indicators warm up

            for (int i = startIdx; i < data.Count; i++)
            {
                var fv = new FeatureVector
                {
                    Timestamp = data[i].Timestamp,
                    Symbol = symbol,

                    // Momentum (13 features)
                    RSI_7 = rsi7[i],
                    RSI_14 = rsi14[i],
                    RSI_21 = rsi21[i],
                    RSI_Momentum = i > 0 ? rsi14[i] - rsi14[i - 1] : 0,
                    MACD = macd[i],
                    MACD_Signal = macdSignal[i],
                    MACD_Histogram = macdHist[i],
                    ROC_5 = roc5[i],
                    ROC_10 = roc10[i],
                    ROC_20 = roc20[i],
                    Stochastic_K = stochK[i],
                    Stochastic_D = stochD[i],
                    Williams_R = williamsR[i],

                    // Trend (15 features)
                    SMA_10 = sma10[i],
                    SMA_20 = sma20[i],
                    SMA_50 = sma50[i],
                    SMA_100 = sma100[i],
                    SMA_200 = sma200[i],
                    EMA_12 = ema12[i],
                    EMA_26 = ema26[i],
                    EMA_9 = ema9[i],
                    EMA_21 = ema21[i],
                    ADX = adx[i],
                    DI_Plus = plusDI[i],
                    DI_Minus = minusDI[i],
                    LinearRegressionSlope = linearSlope[i],
                    HurstExponent = i - startIdx < hurstValues.Length ? hurstValues[i - startIdx] : 0.5,

                    // Volatility (8 features)
                    ATR_14 = atr14[i],
                    ATR_20 = atr20[i],
                    BollingerBand_Width = bbWidth[i],
                    BollingerBand_Percent = bbPercent[i],
                    ParkinsonVolatility = parkinsonVol[i],
                    GarmanKlassVolatility = parkinsonVol[i] * 0.9, // Simplified
                    GARCH_Volatility = i < garchVol.Length ? garchVol[i] : 0,
                    VolatilityRegime = _garchModel.GetVolatilityRegime(
                        i < garchVol.Length ? System.Math.Sqrt(garchVol[i]) : 0),

                    // Volume (5 features)
                    VolumeRatio = volumes[i] / TechnicalIndicators.SMA(volumes, 20)[i],
                    OBV = obv[i],
                    OBV_Momentum = i > 0 ? obv[i] - obv[i - 1] : 0,
                    VWAP_Deviation = (closes[i] - vwap[i]) / vwap[i],
                    AccumulationDistribution = ad[i],

                    // Structural (7 features)
                    SupportProximity = CalculateSupportProximity(closes, i),
                    ResistanceProximity = CalculateResistanceProximity(closes, i),
                    GapSize = i > 0 ? (opens[i] - closes[i - 1]) / closes[i - 1] : 0,
                    DayOfWeek = (int)data[i].Timestamp.DayOfWeek,
                    DayOfMonth = data[i].Timestamp.Day,
                    Month = data[i].Timestamp.Month,
                    Quarter = (data[i].Timestamp.Month - 1) / 3 + 1,

                    // Derived (4 features)
                    KalmanPrice = kalmanResults[i].price,
                    KalmanVelocity = kalmanResults[i].velocity,
                    KalmanAcceleration = kalmanResults[i].acceleration,
                    MarketRegime = HurstExponent.GetRegime(
                        i - startIdx < hurstValues.Length ? hurstValues[i - startIdx] : 0.5),

                    // Targets (for training)
                    FutureReturn_1D = i + 1 < closes.Length ? (closes[i + 1] - closes[i]) / closes[i] : null,
                    FutureReturn_5D = i + 5 < closes.Length ? (closes[i + 5] - closes[i]) / closes[i] : null,
                    FutureReturn_20D = i + 20 < closes.Length ? (closes[i + 20] - closes[i]) / closes[i] : null,
                    Direction_1D = i + 1 < closes.Length
                        ? System.Math.Sign(closes[i + 1] - closes[i])
                        : null
                };

                features.Add(fv);
            }

            return features;
        }, cancellationToken);
    }

    public async Task<FeatureVector> ComputeFeaturesOnlineAsync(
        OHLCV latest,
        List<OHLCV> historicalContext,
        string symbol,
        CancellationToken cancellationToken = default)
    {
        // Append latest to context and compute
        var fullData = new List<OHLCV>(historicalContext) { latest };
        var features = await ComputeFeaturesAsync(fullData, symbol, cancellationToken);
        return features.Last();
    }

    public void UpdateFeatureImportance(Dictionary<string, double> importanceScores)
    {
        _featureImportance = new Dictionary<string, double>(importanceScores);
    }

    private static double CalculateSupportProximity(double[] closes, int index, int lookback = 50)
    {
        if (index < lookback) return 0;

        var window = closes.Skip(index - lookback).Take(lookback).ToArray();
        var support = window.Min();
        return (closes[index] - support) / closes[index];
    }

    private static double CalculateResistanceProximity(double[] closes, int index, int lookback = 50)
    {
        if (index < lookback) return 0;

        var window = closes.Skip(index - lookback).Take(lookback).ToArray();
        var resistance = window.Max();
        return (resistance - closes[index]) / closes[index];
    }

    private static double[] RollingStd(double[] values, int window)
    {
        var result = new double[values.Length + 1]; // +1 to match original array length

        for (int i = window; i < result.Length; i++)
        {
            var slice = values.Skip(System.Math.Max(0, i - window)).Take(window).ToArray();
            var mean = slice.Average();
            var variance = slice.Select(v => (v - mean) * (v - mean)).Average();
            result[i] = System.Math.Sqrt(variance);
        }

        return result;
    }
}
