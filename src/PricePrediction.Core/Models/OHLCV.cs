namespace PricePrediction.Core.Models;

/// <summary>
/// Open, High, Low, Close, Volume data point
/// </summary>
public class OHLCV
{
    public DateTime Timestamp { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public decimal Open { get; set; }
    public decimal High { get; set; }
    public decimal Low { get; set; }
    public decimal Close { get; set; }
    public long Volume { get; set; }
    public decimal? AdjustedClose { get; set; }

    // Calculated fields
    public decimal TypicalPrice => (High + Low + Close) / 3m;
    public decimal Range => High - Low;
    public decimal BodySize => Math.Abs(Close - Open);
    public bool IsBullish => Close > Open;
    public bool IsBearish => Close < Open;
}
