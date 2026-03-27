"""Tests for utils/technical_indicators.py — RSI, MACD, Bollinger Bands, signal classifier."""

import math

import pytest

from utils.technical_indicators import (
    calculate_bollinger,
    calculate_macd,
    calculate_rsi,
    classify_signal,
)

# ════════════════════════════════════════════════════════════════
#  RSI Tests
# ════════════════════════════════════════════════════════════════


class TestCalculateRSI:
    """Tests for calculate_rsi()."""

    def test_rsi_returns_float(self):
        """RSI should return a float."""
        prices = [44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28]
        result = calculate_rsi(prices)
        assert isinstance(result, float)

    def test_rsi_range_0_to_100(self):
        """RSI must always be between 0 and 100."""
        prices = [44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28]
        result = calculate_rsi(prices)
        assert 0.0 <= result <= 100.0

    def test_rsi_all_gains_near_100(self):
        """If prices only go up, RSI should be near 100."""
        prices = list(range(100, 120))  # 20 consecutive gains
        result = calculate_rsi(prices)
        assert result > 95.0

    def test_rsi_all_losses_near_0(self):
        """If prices only go down, RSI should be near 0."""
        prices = list(range(120, 100, -1))  # 20 consecutive losses
        result = calculate_rsi(prices)
        assert result < 5.0

    def test_rsi_known_value(self):
        """Verify RSI against a known hand-calculated value.

        Using Wilder's smoothed RSI (period=14):
        Prices from Investopedia RSI example.
        """
        # Classic Investopedia RSI example data (15 prices → 14 changes)
        prices = [
            44.0,
            44.34,
            44.09,
            43.61,
            44.33,
            44.83,
            45.10,
            45.42,
            45.84,
            46.08,
            45.89,
            46.03,
            45.61,
            46.28,
            46.28,
        ]
        result = calculate_rsi(prices, period=14)
        # With 15 prices and period=14, only the initial SMA applies (no smoothing).
        # Avg gain = 3.62/14 ≈ 0.2586, Avg loss = 1.34/14 ≈ 0.0957, RS ≈ 2.70
        # RSI ≈ 100 - 100/(1+2.70) ≈ 73.0
        assert 72.0 < result < 74.0

    def test_rsi_custom_period(self):
        """RSI with a shorter period should still work."""
        prices = [10, 11, 12, 11, 10, 11, 12, 13, 14, 13]
        result = calculate_rsi(prices, period=5)
        assert 0.0 <= result <= 100.0

    def test_rsi_not_enough_data_returns_none(self):
        """If fewer prices than period+1, return None."""
        prices = [100, 101, 102]
        result = calculate_rsi(prices, period=14)
        assert result is None

    def test_rsi_empty_list_returns_none(self):
        """Empty price list should return None."""
        assert calculate_rsi([]) is None

    def test_rsi_flat_prices_returns_50(self):
        """If all prices are the same (no change), RSI should be 50."""
        prices = [100.0] * 20
        result = calculate_rsi(prices)
        assert result == 50.0

    @pytest.mark.parametrize("period", [5, 10, 14, 20])
    def test_rsi_various_periods(self, period):
        """RSI should work with various standard periods."""
        prices = [100 + i * 0.5 * ((-1) ** i) for i in range(period + 10)]
        result = calculate_rsi(prices, period=period)
        assert result is not None
        assert 0.0 <= result <= 100.0


# ════════════════════════════════════════════════════════════════
#  MACD Tests
# ════════════════════════════════════════════════════════════════


class TestCalculateMACD:
    """Tests for calculate_macd()."""

    # Need at least 26 + 9 - 1 = 34 prices for default MACD(12,26,9)
    @pytest.fixture
    def trending_up_prices(self):
        """40 prices with an upward trend."""
        return [100.0 + i * 0.5 for i in range(40)]

    @pytest.fixture
    def trending_down_prices(self):
        """40 prices with a downward trend."""
        return [120.0 - i * 0.5 for i in range(40)]

    @pytest.fixture
    def sideways_prices(self):
        """40 prices oscillating around 100."""
        return [100.0 + (i % 4 - 1.5) for i in range(40)]

    def test_macd_returns_dict(self, trending_up_prices):
        """MACD should return a dict with macd_line, signal_line, histogram."""
        result = calculate_macd(trending_up_prices)
        assert isinstance(result, dict)
        assert "macd_line" in result
        assert "signal_line" in result
        assert "histogram" in result

    def test_macd_values_are_floats(self, trending_up_prices):
        """All MACD output values should be floats."""
        result = calculate_macd(trending_up_prices)
        assert isinstance(result["macd_line"], float)
        assert isinstance(result["signal_line"], float)
        assert isinstance(result["histogram"], float)

    def test_macd_histogram_equals_diff(self, trending_up_prices):
        """Histogram = MACD line - Signal line."""
        result = calculate_macd(trending_up_prices)
        expected = round(result["macd_line"] - result["signal_line"], 4)
        assert result["histogram"] == expected

    def test_macd_uptrend_positive(self, trending_up_prices):
        """In a strong uptrend, MACD line should be positive (fast EMA > slow EMA)."""
        result = calculate_macd(trending_up_prices)
        assert result["macd_line"] > 0

    def test_macd_downtrend_negative(self, trending_down_prices):
        """In a strong downtrend, MACD line should be negative."""
        result = calculate_macd(trending_down_prices)
        assert result["macd_line"] < 0

    def test_macd_sideways_near_zero(self, sideways_prices):
        """In sideways market, MACD should be near zero."""
        result = calculate_macd(sideways_prices)
        assert abs(result["macd_line"]) < 2.0

    def test_macd_not_enough_data_returns_none(self):
        """If fewer prices than slow_period, return None."""
        prices = [100.0] * 10
        result = calculate_macd(prices)
        assert result is None

    def test_macd_empty_list_returns_none(self):
        """Empty price list should return None."""
        assert calculate_macd([]) is None

    def test_macd_custom_periods(self, trending_up_prices):
        """MACD with custom periods should work."""
        result = calculate_macd(trending_up_prices, fast=8, slow=17, signal=9)
        assert result is not None
        assert "macd_line" in result

    def test_macd_flat_prices_zero(self):
        """Flat prices should give MACD ≈ 0."""
        prices = [100.0] * 40
        result = calculate_macd(prices)
        assert abs(result["macd_line"]) < 0.001
        assert abs(result["signal_line"]) < 0.001
        assert abs(result["histogram"]) < 0.001


# ════════════════════════════════════════════════════════════════
#  Bollinger Bands Tests
# ════════════════════════════════════════════════════════════════


class TestCalculateBollinger:
    """Tests for calculate_bollinger()."""

    @pytest.fixture
    def steady_prices(self):
        """25 prices hovering around 100."""
        return [100.0 + (i % 3 - 1) * 0.5 for i in range(25)]

    def test_bollinger_returns_dict(self, steady_prices):
        """Should return dict with upper, middle, lower."""
        result = calculate_bollinger(steady_prices)
        assert isinstance(result, dict)
        assert "upper" in result
        assert "middle" in result
        assert "lower" in result

    def test_bollinger_values_are_floats(self, steady_prices):
        """All values should be floats."""
        result = calculate_bollinger(steady_prices)
        assert isinstance(result["upper"], float)
        assert isinstance(result["middle"], float)
        assert isinstance(result["lower"], float)

    def test_bollinger_upper_above_middle_above_lower(self, steady_prices):
        """upper > middle > lower (always)."""
        result = calculate_bollinger(steady_prices)
        assert result["upper"] > result["middle"]
        assert result["middle"] > result["lower"]

    def test_bollinger_symmetry(self, steady_prices):
        """Upper and lower should be equidistant from middle."""
        result = calculate_bollinger(steady_prices)
        upper_dist = result["upper"] - result["middle"]
        lower_dist = result["middle"] - result["lower"]
        assert abs(upper_dist - lower_dist) < 0.0001

    def test_bollinger_middle_is_sma(self):
        """Middle band should equal the SMA of the last `period` prices."""
        prices = [100.0 + i for i in range(25)]
        result = calculate_bollinger(prices, period=20)
        expected_sma = sum(prices[-20:]) / 20
        assert abs(result["middle"] - expected_sma) < 0.0001

    def test_bollinger_band_width_with_known_std(self):
        """With known std dev, band width should be 2 * num_std * std."""
        prices = [100.0] * 19 + [110.0]  # last price is an outlier
        result = calculate_bollinger(prices, period=20, num_std=2.0)
        width = result["upper"] - result["lower"]
        # Width = 2 * 2.0 * stddev
        mean = sum(prices[-20:]) / 20
        variance = sum((p - mean) ** 2 for p in prices[-20:]) / 20
        std = math.sqrt(variance)
        expected_width = 2 * 2.0 * std
        assert abs(width - expected_width) < 0.0001

    def test_bollinger_flat_prices_tight_bands(self):
        """Flat prices should have very tight bands (std ≈ 0)."""
        prices = [100.0] * 25
        result = calculate_bollinger(prices)
        assert result["upper"] == result["middle"]
        assert result["lower"] == result["middle"]

    def test_bollinger_volatile_prices_wide_bands(self):
        """Volatile prices should have wider bands."""
        calm = [100.0 + (i % 2) * 0.1 for i in range(25)]
        volatile = [100.0 + (i % 2) * 10.0 for i in range(25)]
        calm_result = calculate_bollinger(calm)
        volatile_result = calculate_bollinger(volatile)
        calm_width = calm_result["upper"] - calm_result["lower"]
        volatile_width = volatile_result["upper"] - volatile_result["lower"]
        assert volatile_width > calm_width * 5

    def test_bollinger_not_enough_data_returns_none(self):
        """If fewer prices than period, return None."""
        prices = [100.0] * 5
        result = calculate_bollinger(prices, period=20)
        assert result is None

    def test_bollinger_empty_list_returns_none(self):
        """Empty list should return None."""
        assert calculate_bollinger([]) is None

    def test_bollinger_custom_period_and_std(self):
        """Custom period and num_std should work."""
        prices = [100.0 + i * 0.3 for i in range(15)]
        result = calculate_bollinger(prices, period=10, num_std=1.5)
        assert result is not None
        assert result["upper"] > result["middle"] > result["lower"]


# ════════════════════════════════════════════════════════════════
#  Signal Classifier Tests
# ════════════════════════════════════════════════════════════════

VALID_SIGNALS = {"strong_buy", "buy", "hold", "sell", "strong_sell"}


class TestClassifySignal:
    """Tests for classify_signal()."""

    def test_returns_string(self):
        """Should return a string."""
        result = classify_signal(rsi=50, macd_histogram=0.0, price=100, bb_upper=105, bb_lower=95)
        assert isinstance(result, str)

    def test_returns_valid_signal(self):
        """Return value must be one of the 5 valid signals."""
        result = classify_signal(rsi=50, macd_histogram=0.0, price=100, bb_upper=105, bb_lower=95)
        assert result in VALID_SIGNALS

    def test_strong_buy_oversold_and_bullish(self):
        """RSI < 30 + positive MACD + price near lower BB → strong_buy."""
        result = classify_signal(rsi=22, macd_histogram=0.5, price=95.5, bb_upper=110, bb_lower=95)
        assert result == "strong_buy"

    def test_strong_sell_overbought_and_bearish(self):
        """RSI > 70 + negative MACD + price near upper BB → strong_sell."""
        result = classify_signal(rsi=82, macd_histogram=-0.5, price=109.5, bb_upper=110, bb_lower=90)
        assert result == "strong_sell"

    def test_buy_moderately_bullish(self):
        """RSI < 45 + positive MACD → buy."""
        result = classify_signal(rsi=38, macd_histogram=0.3, price=100, bb_upper=110, bb_lower=90)
        assert result == "buy"

    def test_sell_moderately_bearish(self):
        """RSI > 55 + negative MACD → sell."""
        result = classify_signal(rsi=62, macd_histogram=-0.3, price=100, bb_upper=110, bb_lower=90)
        assert result == "sell"

    def test_hold_neutral_conditions(self):
        """Neutral RSI + small MACD + mid BB → hold."""
        result = classify_signal(rsi=50, macd_histogram=0.0, price=100, bb_upper=105, bb_lower=95)
        assert result == "hold"

    def test_none_rsi_returns_hold(self):
        """If RSI is None (insufficient data), return hold."""
        result = classify_signal(rsi=None, macd_histogram=0.5, price=100, bb_upper=110, bb_lower=90)
        assert result == "hold"

    def test_none_macd_returns_hold(self):
        """If MACD histogram is None, return hold."""
        result = classify_signal(rsi=30, macd_histogram=None, price=100, bb_upper=110, bb_lower=90)
        assert result == "hold"

    def test_none_bb_returns_hold(self):
        """If BB values are None, return hold."""
        result = classify_signal(rsi=30, macd_histogram=0.5, price=100, bb_upper=None, bb_lower=None)
        assert result == "hold"

    @pytest.mark.parametrize(
        "rsi,macd_h,price,bb_up,bb_low",
        [
            (25, 1.0, 91, 110, 90),  # strong_buy signals
            (75, -1.0, 109, 110, 90),  # strong_sell signals
            (40, 0.5, 100, 110, 90),  # buy signals
            (60, -0.5, 100, 110, 90),  # sell signals
            (50, 0.0, 100, 110, 90),  # hold
        ],
    )
    def test_always_returns_valid_signal(self, rsi, macd_h, price, bb_up, bb_low):
        """Every combination must return a valid signal."""
        result = classify_signal(rsi=rsi, macd_histogram=macd_h, price=price, bb_upper=bb_up, bb_lower=bb_low)
        assert result in VALID_SIGNALS
