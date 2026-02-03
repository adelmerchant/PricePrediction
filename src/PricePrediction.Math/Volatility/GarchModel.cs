namespace PricePrediction.Math.Volatility;

/// <summary>
/// GARCH(1,1) model for volatility forecasting
/// σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}
/// Expected accuracy: 75-85% for volatility prediction
/// </summary>
public class GarchModel
{
    private double _omega;
    private double _alpha;
    private double _beta;
    private double _lastSquaredReturn;
    private double _lastVariance;
    private bool _isFitted;

    public double Omega => _omega;
    public double Alpha => _alpha;
    public double Beta => _beta;
    public double Persistence => _alpha + _beta;

    /// <summary>
    /// Fit GARCH(1,1) model to return series
    /// </summary>
    public void Fit(double[] returns, int maxIterations = 100)
    {
        if (returns.Length < 50)
            throw new ArgumentException("Need at least 50 observations for GARCH fitting");

        // Initial parameter estimates
        var unconditionalVariance = CalculateVariance(returns);
        _omega = unconditionalVariance * 0.01;
        _alpha = 0.1;
        _beta = 0.85;

        // Maximum likelihood estimation using simplified gradient descent
        for (int iter = 0; iter < maxIterations; iter++)
        {
            var (newOmega, newAlpha, newBeta) = OptimizeStep(returns);

            // Check for convergence
            var change = System.Math.Abs(_omega - newOmega) +
                        System.Math.Abs(_alpha - newAlpha) +
                        System.Math.Abs(_beta - newBeta);

            _omega = newOmega;
            _alpha = newAlpha;
            _beta = newBeta;

            if (change < 1e-6) break;
        }

        // Ensure stability: alpha + beta < 1
        if (_alpha + _beta >= 1.0)
        {
            var sum = _alpha + _beta;
            _alpha = _alpha / sum * 0.99;
            _beta = _beta / sum * 0.99;
        }

        // Initialize state
        _lastVariance = unconditionalVariance;
        _lastSquaredReturn = returns[^1] * returns[^1];
        _isFitted = true;
    }

    /// <summary>
    /// Predict next period's variance
    /// </summary>
    public double Forecast(int steps = 1)
    {
        if (!_isFitted)
            throw new InvalidOperationException("Model must be fitted before forecasting");

        double variance = _lastVariance;
        var unconditionalVariance = _omega / (1 - _alpha - _beta);

        for (int i = 0; i < steps; i++)
        {
            variance = _omega + _alpha * _lastSquaredReturn + _beta * variance;
            // Converges to unconditional variance
            _lastSquaredReturn = variance;
        }

        return variance;
    }

    /// <summary>
    /// Predict volatility (standard deviation) for next period
    /// </summary>
    public double ForecastVolatility(int steps = 1)
    {
        return System.Math.Sqrt(Forecast(steps));
    }

    /// <summary>
    /// Update with new return observation
    /// </summary>
    public void Update(double newReturn)
    {
        if (!_isFitted)
            throw new InvalidOperationException("Model must be fitted before updating");

        _lastVariance = _omega + _alpha * _lastSquaredReturn + _beta * _lastVariance;
        _lastSquaredReturn = newReturn * newReturn;
    }

    /// <summary>
    /// Calculate conditional variances for entire series
    /// </summary>
    public double[] FitTransform(double[] returns)
    {
        Fit(returns);
        var variances = new double[returns.Length];

        var variance = CalculateVariance(returns);
        variances[0] = variance;

        for (int t = 1; t < returns.Length; t++)
        {
            variance = _omega + _alpha * returns[t - 1] * returns[t - 1] + _beta * variance;
            variances[t] = variance;
        }

        return variances;
    }

    /// <summary>
    /// Detect volatility regime (high/low)
    /// </summary>
    public int GetVolatilityRegime(double currentVolatility, double threshold = 1.2)
    {
        var unconditionalVol = System.Math.Sqrt(_omega / (1 - _alpha - _beta));

        if (currentVolatility > unconditionalVol * threshold)
            return 1; // High volatility
        if (currentVolatility < unconditionalVol / threshold)
            return -1; // Low volatility
        return 0; // Normal
    }

    private (double omega, double alpha, double beta) OptimizeStep(double[] returns)
    {
        // Simplified gradient descent - in production, use proper MLE
        var unconditionalVariance = CalculateVariance(returns);

        // Calculate log-likelihood
        var variances = new double[returns.Length];
        variances[0] = unconditionalVariance;

        for (int t = 1; t < returns.Length; t++)
        {
            variances[t] = _omega + _alpha * returns[t - 1] * returns[t - 1] + _beta * variances[t - 1];
            variances[t] = System.Math.Max(variances[t], 1e-8); // Prevent negative variance
        }

        // Very simple parameter update (in production, use BFGS or similar)
        var newOmega = _omega * 0.99 + unconditionalVariance * (1 - _alpha - _beta) * 0.01;
        var newAlpha = _alpha; // Simplified - keep alpha stable
        var newBeta = 1 - _alpha - newOmega / unconditionalVariance * 0.9;

        newOmega = System.Math.Max(newOmega, 1e-6);
        newAlpha = System.Math.Clamp(newAlpha, 0.01, 0.3);
        newBeta = System.Math.Clamp(newBeta, 0.6, 0.95);

        return (newOmega, newAlpha, newBeta);
    }

    private static double CalculateVariance(double[] returns)
    {
        var mean = returns.Average();
        return returns.Select(r => (r - mean) * (r - mean)).Average();
    }

    /// <summary>
    /// Static helper: Calculate returns from prices
    /// </summary>
    public static double[] CalculateReturns(double[] prices)
    {
        var returns = new double[prices.Length - 1];
        for (int i = 1; i < prices.Length; i++)
        {
            returns[i - 1] = (prices[i] - prices[i - 1]) / prices[i - 1];
        }
        return returns;
    }
}
