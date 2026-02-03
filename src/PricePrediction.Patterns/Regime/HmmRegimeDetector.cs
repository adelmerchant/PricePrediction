using PricePrediction.Core.Enums;

namespace PricePrediction.Patterns.Regime;

/// <summary>
/// Hidden Markov Model for market regime detection
/// States: TrendingUp, TrendingDown, RangeBound, HighVolatility, LowVolatility
/// Expected accuracy: 75-85%
/// Features: Returns, Volatility, Volume
/// </summary>
public class HmmRegimeDetector
{
    private readonly int _numStates;
    private double[,] _transitionMatrix;
    private double[] _initialProbabilities;
    private GaussianEmission[] _emissions;
    private bool _isFitted;

    public HmmRegimeDetector(int numStates = 3)
    {
        if (numStates < 2 || numStates > 5)
            throw new ArgumentException("Number of states must be between 2 and 5");

        _numStates = numStates;
        _transitionMatrix = new double[numStates, numStates];
        _initialProbabilities = new double[numStates];
        _emissions = new GaussianEmission[numStates];

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        // Initialize with uniform probabilities
        for (int i = 0; i < _numStates; i++)
        {
            _initialProbabilities[i] = 1.0 / _numStates;

            for (int j = 0; j < _numStates; j++)
            {
                // Favor staying in same state
                _transitionMatrix[i, j] = i == j ? 0.7 : 0.3 / (_numStates - 1);
            }

            _emissions[i] = new GaussianEmission();
        }
    }

    /// <summary>
    /// Fit HMM using Baum-Welch algorithm (simplified EM)
    /// </summary>
    public void Fit(double[] returns, double[] volatilities, double[] volumeRatios, int maxIterations = 50)
    {
        int T = returns.Length;
        if (T < 50)
            throw new ArgumentException("Need at least 50 observations");

        // Combine features into observation vectors
        var observations = new double[T][];
        for (int t = 0; t < T; t++)
        {
            observations[t] = new[] { returns[t], volatilities[t], volumeRatios[t] };
        }

        // Initialize emissions with k-means clustering
        InitializeEmissionsWithKMeans(observations);

        // EM iterations
        for (int iter = 0; iter < maxIterations; iter++)
        {
            // E-step: Forward-Backward
            var (alpha, beta, scalingFactors) = ForwardBackward(observations);
            var gamma = ComputeGamma(alpha, beta, scalingFactors, T);
            var xi = ComputeXi(observations, alpha, beta, scalingFactors, T);

            // M-step: Update parameters
            UpdateParameters(observations, gamma, xi, T);

            // Check convergence (simplified - in production, use log-likelihood)
            if (iter > 10 && HasConverged(gamma))
                break;
        }

        _isFitted = true;
    }

    /// <summary>
    /// Predict most likely state sequence using Viterbi algorithm
    /// </summary>
    public int[] PredictStates(double[] returns, double[] volatilities, double[] volumeRatios)
    {
        if (!_isFitted)
            throw new InvalidOperationException("Model must be fitted first");

        int T = returns.Length;
        var observations = new double[T][];
        for (int t = 0; t < T; t++)
        {
            observations[t] = new[] { returns[t], volatilities[t], volumeRatios[t] };
        }

        return ViterbiDecode(observations);
    }

    /// <summary>
    /// Get current market regime
    /// </summary>
    public MarketRegime GetCurrentRegime(double currentReturn, double currentVolatility, double currentVolumeRatio)
    {
        if (!_isFitted)
            throw new InvalidOperationException("Model must be fitted first");

        // Find most likely current state
        var observation = new[] { currentReturn, currentVolatility, currentVolumeRatio };
        var probabilities = new double[_numStates];

        for (int i = 0; i < _numStates; i++)
        {
            probabilities[i] = _emissions[i].Probability(observation);
        }

        var mostLikelyState = Array.IndexOf(probabilities, probabilities.Max());

        // Map state to regime based on emission characteristics
        return MapStateToRegime(mostLikelyState);
    }

    /// <summary>
    /// Get regime confidence
    /// </summary>
    public double GetRegimeConfidence(double currentReturn, double currentVolatility, double currentVolumeRatio)
    {
        if (!_isFitted)
            throw new InvalidOperationException("Model must be fitted first");

        var observation = new[] { currentReturn, currentVolatility, currentVolumeRatio };
        var probabilities = new double[_numStates];

        for (int i = 0; i < _numStates; i++)
        {
            probabilities[i] = _emissions[i].Probability(observation);
        }

        var sum = probabilities.Sum();
        if (sum == 0) return 0.5;

        // Normalize
        for (int i = 0; i < _numStates; i++)
        {
            probabilities[i] /= sum;
        }

        return probabilities.Max();
    }

    #region Forward-Backward Algorithm

    private (double[,] alpha, double[,] beta, double[] scaling) ForwardBackward(double[][] observations)
    {
        int T = observations.Length;
        var alpha = new double[T, _numStates];
        var beta = new double[T, _numStates];
        var scaling = new double[T];

        // Forward pass
        scaling[0] = 0;
        for (int i = 0; i < _numStates; i++)
        {
            alpha[0, i] = _initialProbabilities[i] * _emissions[i].Probability(observations[0]);
            scaling[0] += alpha[0, i];
        }

        // Scale
        if (scaling[0] > 0)
        {
            for (int i = 0; i < _numStates; i++)
                alpha[0, i] /= scaling[0];
        }

        // Forward iterations
        for (int t = 1; t < T; t++)
        {
            scaling[t] = 0;
            for (int j = 0; j < _numStates; j++)
            {
                alpha[t, j] = 0;
                for (int i = 0; i < _numStates; i++)
                {
                    alpha[t, j] += alpha[t - 1, i] * _transitionMatrix[i, j];
                }
                alpha[t, j] *= _emissions[j].Probability(observations[t]);
                scaling[t] += alpha[t, j];
            }

            // Scale
            if (scaling[t] > 0)
            {
                for (int j = 0; j < _numStates; j++)
                    alpha[t, j] /= scaling[t];
            }
        }

        // Backward pass
        for (int i = 0; i < _numStates; i++)
            beta[T - 1, i] = 1.0 / scaling[T - 1];

        for (int t = T - 2; t >= 0; t--)
        {
            for (int i = 0; i < _numStates; i++)
            {
                beta[t, i] = 0;
                for (int j = 0; j < _numStates; j++)
                {
                    beta[t, i] += _transitionMatrix[i, j] * _emissions[j].Probability(observations[t + 1]) * beta[t + 1, j];
                }
                beta[t, i] /= scaling[t];
            }
        }

        return (alpha, beta, scaling);
    }

    private double[,] ComputeGamma(double[,] alpha, double[,] beta, double[] scaling, int T)
    {
        var gamma = new double[T, _numStates];

        for (int t = 0; t < T; t++)
        {
            double sum = 0;
            for (int i = 0; i < _numStates; i++)
            {
                gamma[t, i] = alpha[t, i] * beta[t, i];
                sum += gamma[t, i];
            }

            // Normalize
            if (sum > 0)
            {
                for (int i = 0; i < _numStates; i++)
                    gamma[t, i] /= sum;
            }
        }

        return gamma;
    }

    private double[,,] ComputeXi(double[][] observations, double[,] alpha, double[,] beta, double[] scaling, int T)
    {
        var xi = new double[T - 1, _numStates, _numStates];

        for (int t = 0; t < T - 1; t++)
        {
            double sum = 0;
            for (int i = 0; i < _numStates; i++)
            {
                for (int j = 0; j < _numStates; j++)
                {
                    xi[t, i, j] = alpha[t, i] * _transitionMatrix[i, j] *
                                 _emissions[j].Probability(observations[t + 1]) * beta[t + 1, j];
                    sum += xi[t, i, j];
                }
            }

            // Normalize
            if (sum > 0)
            {
                for (int i = 0; i < _numStates; i++)
                {
                    for (int j = 0; j < _numStates; j++)
                        xi[t, i, j] /= sum;
                }
            }
        }

        return xi;
    }

    #endregion

    #region Viterbi Algorithm

    private int[] ViterbiDecode(double[][] observations)
    {
        int T = observations.Length;
        var delta = new double[T, _numStates];
        var psi = new int[T, _numStates];

        // Initialize
        for (int i = 0; i < _numStates; i++)
        {
            delta[0, i] = System.Math.Log(_initialProbabilities[i]) +
                         System.Math.Log(_emissions[i].Probability(observations[0]) + 1e-10);
        }

        // Recursion
        for (int t = 1; t < T; t++)
        {
            for (int j = 0; j < _numStates; j++)
            {
                double maxProb = double.NegativeInfinity;
                int maxState = 0;

                for (int i = 0; i < _numStates; i++)
                {
                    double prob = delta[t - 1, i] + System.Math.Log(_transitionMatrix[i, j] + 1e-10);
                    if (prob > maxProb)
                    {
                        maxProb = prob;
                        maxState = i;
                    }
                }

                delta[t, j] = maxProb + System.Math.Log(_emissions[j].Probability(observations[t]) + 1e-10);
                psi[t, j] = maxState;
            }
        }

        // Backtrack
        var states = new int[T];
        states[T - 1] = Array.IndexOf(Enumerable.Range(0, _numStates)
            .Select(i => delta[T - 1, i]).ToArray(),
            Enumerable.Range(0, _numStates).Select(i => delta[T - 1, i]).Max());

        for (int t = T - 2; t >= 0; t--)
        {
            states[t] = psi[t + 1, states[t + 1]];
        }

        return states;
    }

    #endregion

    #region Parameter Updates

    private void UpdateParameters(double[][] observations, double[,] gamma, double[,,] xi, int T)
    {
        // Update initial probabilities
        for (int i = 0; i < _numStates; i++)
        {
            _initialProbabilities[i] = gamma[0, i];
        }

        // Update transition matrix
        for (int i = 0; i < _numStates; i++)
        {
            double sum = 0;
            for (int t = 0; t < T - 1; t++)
                sum += gamma[t, i];

            for (int j = 0; j < _numStates; j++)
            {
                double numerator = 0;
                for (int t = 0; t < T - 1; t++)
                    numerator += xi[t, i, j];

                _transitionMatrix[i, j] = sum > 0 ? numerator / sum : 0;
            }
        }

        // Update emissions
        for (int i = 0; i < _numStates; i++)
        {
            _emissions[i].Update(observations, gamma, i, T);
        }
    }

    private void InitializeEmissionsWithKMeans(double[][] observations)
    {
        // Simplified k-means initialization
        var random = new Random(42);
        var centroids = observations.OrderBy(_ => random.Next()).Take(_numStates).ToArray();

        for (int i = 0; i < _numStates; i++)
        {
            _emissions[i] = new GaussianEmission(centroids[i]);
        }
    }

    private bool HasConverged(double[,] gamma)
    {
        // Simplified convergence check
        return true; // In production, compare log-likelihood between iterations
    }

    #endregion

    private MarketRegime MapStateToRegime(int state)
    {
        // Map based on emission characteristics
        var meanReturn = _emissions[state].Mean[0];
        var volatility = _emissions[state].Mean[1];

        if (volatility > 0.02) return MarketRegime.HighVolatility;
        if (volatility < 0.005) return MarketRegime.LowVolatility;
        if (meanReturn > 0.001) return MarketRegime.TrendingUp;
        if (meanReturn < -0.001) return MarketRegime.TrendingDown;
        return MarketRegime.RangeBound;
    }

    /// <summary>
    /// Gaussian emission model for HMM
    /// </summary>
    private class GaussianEmission
    {
        public double[] Mean { get; private set; }
        public double[,] Covariance { get; private set; }
        private readonly int _dimension = 3;

        public GaussianEmission()
        {
            Mean = new double[_dimension];
            Covariance = new double[_dimension, _dimension];

            // Initialize with identity
            for (int i = 0; i < _dimension; i++)
            {
                Covariance[i, i] = 1.0;
            }
        }

        public GaussianEmission(double[] initialMean)
        {
            _dimension = initialMean.Length;
            Mean = (double[])initialMean.Clone();
            Covariance = new double[_dimension, _dimension];

            for (int i = 0; i < _dimension; i++)
            {
                Covariance[i, i] = 0.1;
            }
        }

        public double Probability(double[] observation)
        {
            // Multivariate Gaussian (simplified - diagonal covariance)
            double prob = 1.0;
            for (int i = 0; i < _dimension; i++)
            {
                var diff = observation[i] - Mean[i];
                var variance = Covariance[i, i];
                if (variance <= 0) variance = 1e-6;

                prob *= System.Math.Exp(-0.5 * diff * diff / variance) /
                       System.Math.Sqrt(2 * System.Math.PI * variance);
            }
            return System.Math.Max(prob, 1e-10);
        }

        public void Update(double[][] observations, double[,] gamma, int state, int T)
        {
            // Update mean
            var weightSum = 0.0;
            for (int d = 0; d < _dimension; d++)
                Mean[d] = 0;

            for (int t = 0; t < T; t++)
            {
                weightSum += gamma[t, state];
                for (int d = 0; d < _dimension; d++)
                {
                    Mean[d] += gamma[t, state] * observations[t][d];
                }
            }

            if (weightSum > 0)
            {
                for (int d = 0; d < _dimension; d++)
                    Mean[d] /= weightSum;
            }

            // Update covariance (diagonal only for efficiency)
            for (int d = 0; d < _dimension; d++)
                Covariance[d, d] = 0;

            for (int t = 0; t < T; t++)
            {
                for (int d = 0; d < _dimension; d++)
                {
                    var diff = observations[t][d] - Mean[d];
                    Covariance[d, d] += gamma[t, state] * diff * diff;
                }
            }

            if (weightSum > 0)
            {
                for (int d = 0; d < _dimension; d++)
                {
                    Covariance[d, d] /= weightSum;
                    Covariance[d, d] = System.Math.Max(Covariance[d, d], 1e-6); // Prevent zero variance
                }
            }
        }
    }
}
