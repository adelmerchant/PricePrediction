namespace PricePrediction.ML.Indicators;

/// <summary>
/// Technical indicator calculations for feature engineering
/// </summary>
public static class TechnicalIndicators
{
    #region Momentum Indicators

    /// <summary>
    /// Calculate RSI (Relative Strength Index)
    /// </summary>
    public static double[] RSI(double[] prices, int period = 14)
    {
        if (prices.Length < period + 1)
            return Array.Empty<double>();

        var rsi = new double[prices.Length];
        var gains = new double[prices.Length];
        var losses = new double[prices.Length];

        // Calculate price changes
        for (int i = 1; i < prices.Length; i++)
        {
            var change = prices[i] - prices[i - 1];
            gains[i] = change > 0 ? change : 0;
            losses[i] = change < 0 ? -change : 0;
        }

        // Initial average
        var avgGain = gains.Skip(1).Take(period).Average();
        var avgLoss = losses.Skip(1).Take(period).Average();

        for (int i = period; i < prices.Length; i++)
        {
            if (i == period)
            {
                rsi[i] = avgLoss == 0 ? 100 : 100 - (100 / (1 + avgGain / avgLoss));
            }
            else
            {
                avgGain = (avgGain * (period - 1) + gains[i]) / period;
                avgLoss = (avgLoss * (period - 1) + losses[i]) / period;
                rsi[i] = avgLoss == 0 ? 100 : 100 - (100 / (1 + avgGain / avgLoss));
            }
        }

        return rsi;
    }

    /// <summary>
    /// Calculate MACD (Moving Average Convergence Divergence)
    /// Returns: (MACD line, Signal line, Histogram)
    /// </summary>
    public static (double[] macd, double[] signal, double[] histogram) MACD(
        double[] prices, int fastPeriod = 12, int slowPeriod = 26, int signalPeriod = 9)
    {
        var fastEMA = EMA(prices, fastPeriod);
        var slowEMA = EMA(prices, slowPeriod);

        var macd = new double[prices.Length];
        for (int i = 0; i < prices.Length; i++)
        {
            macd[i] = fastEMA[i] - slowEMA[i];
        }

        var signal = EMA(macd, signalPeriod);
        var histogram = new double[prices.Length];
        for (int i = 0; i < prices.Length; i++)
        {
            histogram[i] = macd[i] - signal[i];
        }

        return (macd, signal, histogram);
    }

    /// <summary>
    /// Calculate Rate of Change (ROC)
    /// </summary>
    public static double[] ROC(double[] prices, int period = 10)
    {
        var roc = new double[prices.Length];
        for (int i = period; i < prices.Length; i++)
        {
            roc[i] = ((prices[i] - prices[i - period]) / prices[i - period]) * 100;
        }
        return roc;
    }

    /// <summary>
    /// Calculate Stochastic Oscillator
    /// Returns: (%K, %D)
    /// </summary>
    public static (double[] k, double[] d) Stochastic(
        double[] highs, double[] lows, double[] closes, int period = 14, int smoothK = 3, int smoothD = 3)
    {
        var k = new double[closes.Length];

        for (int i = period - 1; i < closes.Length; i++)
        {
            var highestHigh = highs.Skip(i - period + 1).Take(period).Max();
            var lowestLow = lows.Skip(i - period + 1).Take(period).Min();
            k[i] = highestHigh == lowestLow ? 50 : ((closes[i] - lowestLow) / (highestHigh - lowestLow)) * 100;
        }

        // Smooth %K
        k = SMA(k, smoothK);

        // %D is SMA of %K
        var d = SMA(k, smoothD);

        return (k, d);
    }

    /// <summary>
    /// Calculate Williams %R
    /// </summary>
    public static double[] WilliamsR(double[] highs, double[] lows, double[] closes, int period = 14)
    {
        var williamsR = new double[closes.Length];

        for (int i = period - 1; i < closes.Length; i++)
        {
            var highestHigh = highs.Skip(i - period + 1).Take(period).Max();
            var lowestLow = lows.Skip(i - period + 1).Take(period).Min();
            williamsR[i] = highestHigh == lowestLow ? -50 : ((highestHigh - closes[i]) / (highestHigh - lowestLow)) * -100;
        }

        return williamsR;
    }

    #endregion

    #region Trend Indicators

    /// <summary>
    /// Calculate Simple Moving Average (SMA)
    /// </summary>
    public static double[] SMA(double[] values, int period)
    {
        var sma = new double[values.Length];
        for (int i = period - 1; i < values.Length; i++)
        {
            sma[i] = values.Skip(i - period + 1).Take(period).Average();
        }
        return sma;
    }

    /// <summary>
    /// Calculate Exponential Moving Average (EMA)
    /// </summary>
    public static double[] EMA(double[] values, int period)
    {
        var ema = new double[values.Length];
        var multiplier = 2.0 / (period + 1);

        // Start with SMA
        ema[period - 1] = values.Take(period).Average();

        for (int i = period; i < values.Length; i++)
        {
            ema[i] = (values[i] - ema[i - 1]) * multiplier + ema[i - 1];
        }

        return ema;
    }

    /// <summary>
    /// Calculate ADX (Average Directional Index)
    /// Returns: (ADX, +DI, -DI)
    /// </summary>
    public static (double[] adx, double[] plusDI, double[] minusDI) ADX(
        double[] highs, double[] lows, double[] closes, int period = 14)
    {
        var tr = new double[closes.Length];
        var plusDM = new double[closes.Length];
        var minusDM = new double[closes.Length];

        // Calculate True Range and Directional Movement
        for (int i = 1; i < closes.Length; i++)
        {
            var highLow = highs[i] - lows[i];
            var highClose = System.Math.Abs(highs[i] - closes[i - 1]);
            var lowClose = System.Math.Abs(lows[i] - closes[i - 1]);
            tr[i] = System.Math.Max(highLow, System.Math.Max(highClose, lowClose));

            var upMove = highs[i] - highs[i - 1];
            var downMove = lows[i - 1] - lows[i];

            plusDM[i] = (upMove > downMove && upMove > 0) ? upMove : 0;
            minusDM[i] = (downMove > upMove && downMove > 0) ? downMove : 0;
        }

        // Smooth with Wilder's moving average
        var atr = WildersSmoothing(tr, period);
        var smoothedPlusDM = WildersSmoothing(plusDM, period);
        var smoothedMinusDM = WildersSmoothing(minusDM, period);

        var plusDI = new double[closes.Length];
        var minusDI = new double[closes.Length];
        var dx = new double[closes.Length];

        for (int i = period; i < closes.Length; i++)
        {
            plusDI[i] = atr[i] == 0 ? 0 : (smoothedPlusDM[i] / atr[i]) * 100;
            minusDI[i] = atr[i] == 0 ? 0 : (smoothedMinusDM[i] / atr[i]) * 100;

            var diSum = plusDI[i] + minusDI[i];
            dx[i] = diSum == 0 ? 0 : (System.Math.Abs(plusDI[i] - minusDI[i]) / diSum) * 100;
        }

        var adx = WildersSmoothing(dx, period);

        return (adx, plusDI, minusDI);
    }

    #endregion

    #region Volatility Indicators

    /// <summary>
    /// Calculate ATR (Average True Range)
    /// </summary>
    public static double[] ATR(double[] highs, double[] lows, double[] closes, int period = 14)
    {
        var tr = new double[closes.Length];

        for (int i = 1; i < closes.Length; i++)
        {
            var highLow = highs[i] - lows[i];
            var highClose = System.Math.Abs(highs[i] - closes[i - 1]);
            var lowClose = System.Math.Abs(lows[i] - closes[i - 1]);
            tr[i] = System.Math.Max(highLow, System.Math.Max(highClose, lowClose));
        }

        return WildersSmoothing(tr, period);
    }

    /// <summary>
    /// Calculate Bollinger Bands
    /// Returns: (Upper, Middle, Lower, Width, %B)
    /// </summary>
    public static (double[] upper, double[] middle, double[] lower, double[] width, double[] percentB) BollingerBands(
        double[] prices, int period = 20, double stdDevMultiplier = 2.0)
    {
        var middle = SMA(prices, period);
        var upper = new double[prices.Length];
        var lower = new double[prices.Length];
        var width = new double[prices.Length];
        var percentB = new double[prices.Length];

        for (int i = period - 1; i < prices.Length; i++)
        {
            var slice = prices.Skip(i - period + 1).Take(period).ToArray();
            var stdDev = StandardDeviation(slice);

            upper[i] = middle[i] + stdDevMultiplier * stdDev;
            lower[i] = middle[i] - stdDevMultiplier * stdDev;
            width[i] = (upper[i] - lower[i]) / middle[i];
            percentB[i] = upper[i] == lower[i] ? 0.5 : (prices[i] - lower[i]) / (upper[i] - lower[i]);
        }

        return (upper, middle, lower, width, percentB);
    }

    /// <summary>
    /// Parkinson volatility (uses high-low range)
    /// </summary>
    public static double[] ParkinsonVolatility(double[] highs, double[] lows, int period = 20)
    {
        var volatility = new double[highs.Length];

        for (int i = period - 1; i < highs.Length; i++)
        {
            double sum = 0;
            for (int j = i - period + 1; j <= i; j++)
            {
                var logRatio = System.Math.Log(highs[j] / lows[j]);
                sum += logRatio * logRatio;
            }
            volatility[i] = System.Math.Sqrt(sum / (4 * period * System.Math.Log(2)));
        }

        return volatility;
    }

    #endregion

    #region Volume Indicators

    /// <summary>
    /// Calculate OBV (On-Balance Volume)
    /// </summary>
    public static double[] OBV(double[] closes, double[] volumes)
    {
        var obv = new double[closes.Length];
        obv[0] = volumes[0];

        for (int i = 1; i < closes.Length; i++)
        {
            if (closes[i] > closes[i - 1])
                obv[i] = obv[i - 1] + volumes[i];
            else if (closes[i] < closes[i - 1])
                obv[i] = obv[i - 1] - volumes[i];
            else
                obv[i] = obv[i - 1];
        }

        return obv;
    }

    /// <summary>
    /// Calculate VWAP (Volume Weighted Average Price)
    /// </summary>
    public static double[] VWAP(double[] highs, double[] lows, double[] closes, double[] volumes)
    {
        var vwap = new double[closes.Length];
        double cumulativeTPV = 0;
        double cumulativeVolume = 0;

        for (int i = 0; i < closes.Length; i++)
        {
            var typicalPrice = (highs[i] + lows[i] + closes[i]) / 3;
            cumulativeTPV += typicalPrice * volumes[i];
            cumulativeVolume += volumes[i];
            vwap[i] = cumulativeVolume == 0 ? 0 : cumulativeTPV / cumulativeVolume;
        }

        return vwap;
    }

    /// <summary>
    /// Calculate Accumulation/Distribution Line
    /// </summary>
    public static double[] AccumulationDistribution(double[] highs, double[] lows, double[] closes, double[] volumes)
    {
        var ad = new double[closes.Length];

        for (int i = 0; i < closes.Length; i++)
        {
            var clv = (highs[i] == lows[i]) ? 0 : ((closes[i] - lows[i]) - (highs[i] - closes[i])) / (highs[i] - lows[i]);
            var moneyFlowVolume = clv * volumes[i];
            ad[i] = (i == 0 ? 0 : ad[i - 1]) + moneyFlowVolume;
        }

        return ad;
    }

    #endregion

    #region Helper Methods

    private static double[] WildersSmoothing(double[] values, int period)
    {
        var smoothed = new double[values.Length];

        // Initial average
        smoothed[period] = values.Skip(1).Take(period).Average();

        for (int i = period + 1; i < values.Length; i++)
        {
            smoothed[i] = (smoothed[i - 1] * (period - 1) + values[i]) / period;
        }

        return smoothed;
    }

    private static double StandardDeviation(double[] values)
    {
        var mean = values.Average();
        var sumOfSquares = values.Select(v => (v - mean) * (v - mean)).Sum();
        return System.Math.Sqrt(sumOfSquares / values.Length);
    }

    /// <summary>
    /// Linear regression slope
    /// </summary>
    public static double[] LinearRegressionSlope(double[] values, int period = 20)
    {
        var slopes = new double[values.Length];

        for (int i = period - 1; i < values.Length; i++)
        {
            var slice = values.Skip(i - period + 1).Take(period).ToArray();
            slopes[i] = CalculateSlope(slice);
        }

        return slopes;
    }

    private static double CalculateSlope(double[] values)
    {
        var n = values.Length;
        var sumX = 0.0;
        var sumY = 0.0;
        var sumXY = 0.0;
        var sumX2 = 0.0;

        for (int i = 0; i < n; i++)
        {
            sumX += i;
            sumY += values[i];
            sumXY += i * values[i];
            sumX2 += i * i;
        }

        return (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    }

    #endregion
}
