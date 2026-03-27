"""MarketMind AI — Technical indicator calculations.

Pure-Python UDFs for RSI, MACD, Bollinger Bands, and signal classification.
Designed to work as Spark UDFs when registered in Databricks.
"""


def calculate_rsi(prices: list[float], period: int = 14) -> float | None:
    """Calculate Relative Strength Index using Wilder's smoothing.

    Args:
        prices: Chronological list of closing prices.
        period: Lookback period (default 14).

    Returns:
        RSI value (0-100), or None if insufficient data.
    """
    if not prices or len(prices) < period + 1:
        return None

    # Calculate price changes
    changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

    # First average: simple mean of first `period` changes
    gains = [max(0.0, c) for c in changes[:period]]
    losses = [max(0.0, -c) for c in changes[:period]]

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    # Wilder's smoothing for remaining changes
    for c in changes[period:]:
        avg_gain = (avg_gain * (period - 1) + max(0.0, c)) / period
        avg_loss = (avg_loss * (period - 1) + max(0.0, -c)) / period

    # Handle flat prices (no gains, no losses)
    if avg_gain == 0.0 and avg_loss == 0.0:
        return 50.0

    # Handle all losses (avoid division by zero)
    if avg_loss == 0.0:
        return 100.0

    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 4)


def _ema(values: list[float], period: int) -> list[float]:
    """Calculate Exponential Moving Average series.

    First value is SMA of the first `period` values.
    Subsequent values use the EMA formula: EMA = price * k + prev_ema * (1-k).
    """
    if len(values) < period:
        return []

    k = 2.0 / (period + 1)
    sma = sum(values[:period]) / period
    result = [sma]

    for price in values[period:]:
        result.append(price * k + result[-1] * (1 - k))

    return result


def calculate_macd(
    prices: list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict | None:
    """Calculate MACD (Moving Average Convergence Divergence).

    Args:
        prices: Chronological list of closing prices.
        fast: Fast EMA period (default 12).
        slow: Slow EMA period (default 26).
        signal: Signal line EMA period (default 9).

    Returns:
        Dict with macd_line, signal_line, histogram — or None if insufficient data.
    """
    if not prices or len(prices) < slow:
        return None

    fast_ema = _ema(prices, fast)
    slow_ema = _ema(prices, slow)

    # Align: fast_ema has more values than slow_ema.
    # Trim fast_ema to match slow_ema length, aligning from the end.
    offset = len(fast_ema) - len(slow_ema)
    fast_aligned = fast_ema[offset:]

    # MACD line = fast EMA - slow EMA
    macd_series = [f - s for f, s in zip(fast_aligned, slow_ema)]

    # Signal line = EMA of MACD series
    if len(macd_series) < signal:
        # Not enough MACD values for signal line — return latest MACD only
        return {
            "macd_line": round(macd_series[-1], 4),
            "signal_line": round(macd_series[-1], 4),
            "histogram": 0.0,
        }

    signal_ema = _ema(macd_series, signal)
    macd_line = round(macd_series[-1], 4)
    signal_line = round(signal_ema[-1], 4)
    histogram = round(macd_line - signal_line, 4)

    return {
        "macd_line": macd_line,
        "signal_line": signal_line,
        "histogram": histogram,
    }


def calculate_bollinger(
    prices: list[float],
    period: int = 20,
    num_std: float = 2.0,
) -> dict | None:
    """Calculate Bollinger Bands.

    Args:
        prices: Chronological list of closing prices.
        period: SMA lookback period (default 20).
        num_std: Number of standard deviations for bands (default 2.0).

    Returns:
        Dict with upper, middle, lower band values — or None if insufficient data.
    """
    if not prices or len(prices) < period:
        return None

    window = prices[-period:]
    middle = sum(window) / period

    # Population standard deviation (not sample) — matches trading convention
    variance = sum((p - middle) ** 2 for p in window) / period
    std = variance**0.5

    band_offset = num_std * std

    return {
        "upper": round(middle + band_offset, 4),
        "middle": round(middle, 4),
        "lower": round(middle - band_offset, 4),
    }


def classify_signal(
    rsi: float | None,
    macd_histogram: float | None,
    price: float | None,
    bb_upper: float | None,
    bb_lower: float | None,
) -> str:
    """Classify a trading signal based on technical indicators.

    Combines RSI, MACD histogram, and Bollinger Band position into a
    single signal: strong_buy / buy / hold / sell / strong_sell.

    Any None input causes a conservative "hold" return.

    Args:
        rsi: RSI value (0-100).
        macd_histogram: MACD histogram value (positive = bullish).
        price: Current price.
        bb_upper: Bollinger upper band.
        bb_lower: Bollinger lower band.

    Returns:
        One of: "strong_buy", "buy", "hold", "sell", "strong_sell".
    """
    # Insufficient data → hold
    if any(v is None for v in (rsi, macd_histogram, price, bb_upper, bb_lower)):
        return "hold"

    # Score-based system: accumulate bullish (+) and bearish (-) points
    score = 0

    # RSI component (-2 to +2)
    if rsi < 30:
        score += 2  # oversold → bullish
    elif rsi < 45:
        score += 1
    elif rsi > 70:
        score -= 2  # overbought → bearish
    elif rsi > 55:
        score -= 1

    # MACD histogram component (-1 to +1)
    if macd_histogram > 0:
        score += 1
    elif macd_histogram < 0:
        score -= 1

    # Bollinger Band position component (-1 to +1)
    bb_range = bb_upper - bb_lower
    if bb_range > 0:
        bb_position = (price - bb_lower) / bb_range
        if bb_position < 0.15:
            score += 1  # near lower band → bullish
        elif bb_position > 0.85:
            score -= 1  # near upper band → bearish

    # Map score to signal
    if score >= 3:
        return "strong_buy"
    elif score >= 1:
        return "buy"
    elif score <= -3:
        return "strong_sell"
    elif score <= -1:
        return "sell"
    return "hold"
