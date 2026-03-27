"""MarketMind AI — Agent tools for querying market data.

Pure-Python functions that take pre-queried data (lists of dicts)
and return formatted results. No Spark dependency — testable locally.
The Databricks notebook handles the SQL queries and passes data in.
"""


def get_stock_price(daily_records: list[dict], symbol: str) -> dict:
    """Get latest price, change, and volume for a stock symbol.

    Args:
        daily_records: List of daily summary dicts with keys:
            symbol, date, sector, day_open, day_high, day_low,
            day_close, total_volume, avg_change_pct.
        symbol: Stock ticker to look up.

    Returns:
        Dict with price data, or {"error": "..."} if not found.
    """
    symbol = symbol.upper().strip()

    # Filter to this symbol and get the latest date
    matches = [r for r in daily_records if r["symbol"] == symbol]
    if not matches:
        return {"error": f"No data found for {symbol}"}

    latest = max(matches, key=lambda r: r["date"])

    return {
        "symbol": latest["symbol"],
        "date": str(latest["date"]),
        "price": round(latest["day_close"], 2),
        "open": round(latest["day_open"], 2),
        "high": round(latest["day_high"], 2),
        "low": round(latest["day_low"], 2),
        "volume": latest["total_volume"],
        "change_pct": round(latest["avg_change_pct"], 2),
        "sector": latest["sector"],
    }


def get_sector_performance(
    daily_records: list[dict],
    rankings: list[dict],
    sector: str,
) -> dict:
    """Get performance metrics for a sector — top gainers, losers, rankings.

    Args:
        daily_records: Daily summary dicts (must include sector, symbol, avg_change_pct).
        rankings: Sector ranking dicts (must include sector, symbol, sector_rank, percentile).
        sector: Sector name to look up.

    Returns:
        Dict with sector stats, or {"error": "..."} if not found.
    """
    sector = sector.strip().title()

    sector_records = [r for r in daily_records if r.get("sector") == sector]
    if not sector_records:
        return {"error": f"No data found for sector '{sector}'"}

    sector_ranks = [r for r in rankings if r.get("sector") == sector]
    rank_by_symbol = {r["symbol"]: r for r in sector_ranks}

    # Build stocks list sorted by rank
    stocks = []
    for rec in sector_records:
        sym = rec["symbol"]
        rank_info = rank_by_symbol.get(sym, {})
        stocks.append(
            {
                "symbol": sym,
                "change_pct": round(rec["avg_change_pct"], 2),
                "rank": rank_info.get("sector_rank"),
                "percentile": rank_info.get("percentile"),
            }
        )

    stocks.sort(key=lambda s: s["rank"] if s["rank"] is not None else 999)

    return {
        "sector": sector,
        "date": str(sector_records[0]["date"]),
        "stock_count": len(stocks),
        "top_gainer": stocks[0]["symbol"],
        "top_loser": stocks[-1]["symbol"],
        "stocks": stocks,
    }


def get_market_sentiment(
    symbol_sentiments: list[dict],
    market_mood: dict | None,
    symbol: str | None = None,
) -> dict:
    """Get news sentiment — per symbol or overall market mood.

    Args:
        symbol_sentiments: List of per-symbol sentiment dicts with keys:
            symbol, avg_sentiment, article_count, most_positive, most_negative.
        market_mood: Overall market mood dict with keys:
            mood, avg_score, article_count, positive_count, negative_count, neutral_count.
        symbol: If provided, return sentiment for this symbol. Otherwise return market mood.

    Returns:
        Dict with sentiment data, or {"error": "..."} if not found.
    """
    if symbol:
        symbol = symbol.upper().strip()
        match = next((r for r in symbol_sentiments if r["symbol"] == symbol), None)
        if not match:
            return {"error": f"No sentiment data for {symbol}"}

        score = match["avg_sentiment"]
        if score > 0.05:
            label = "bullish"
        elif score < -0.05:
            label = "bearish"
        else:
            label = "neutral"

        return {
            "symbol": match["symbol"],
            "avg_sentiment": round(score, 3),
            "label": label,
            "article_count": match["article_count"],
            "most_positive": match["most_positive"],
            "most_negative": match["most_negative"],
        }

    # No symbol — return overall market mood
    if not market_mood:
        return {"error": "No market mood data available"}

    return {
        "mood": market_mood["mood"],
        "avg_score": round(market_mood["avg_score"], 3),
        "article_count": market_mood["article_count"],
        "positive_count": market_mood["positive_count"],
        "negative_count": market_mood["negative_count"],
        "neutral_count": market_mood["neutral_count"],
    }


def get_technical_signals(signal_records: list[dict], symbol: str) -> dict:
    """Get RSI, MACD, Bollinger Bands, and buy/sell signal for a stock.

    Args:
        signal_records: List of market signal dicts with keys:
            symbol, sector, latest_date, latest_close, rsi, macd_line,
            macd_signal, macd_histogram, bb_upper, bb_middle, bb_lower, signal.
        symbol: Stock ticker to look up.

    Returns:
        Dict with technical data, or {"error": "..."} if not found.
    """
    symbol = symbol.upper().strip()

    match = next((r for r in signal_records if r["symbol"] == symbol), None)
    if not match:
        return {"error": f"No technical data for {symbol}"}

    return {
        "symbol": match["symbol"],
        "sector": match["sector"],
        "date": str(match["latest_date"]),
        "price": round(match["latest_close"], 2),
        "rsi": round(match["rsi"], 2) if match["rsi"] is not None else None,
        "macd": {
            "line": round(match["macd_line"], 4) if match["macd_line"] is not None else None,
            "signal": round(match["macd_signal"], 4) if match["macd_signal"] is not None else None,
            "histogram": round(match["macd_histogram"], 4) if match["macd_histogram"] is not None else None,
        },
        "bollinger": {
            "upper": round(match["bb_upper"], 2) if match["bb_upper"] is not None else None,
            "middle": round(match["bb_middle"], 2) if match["bb_middle"] is not None else None,
            "lower": round(match["bb_lower"], 2) if match["bb_lower"] is not None else None,
        },
        "signal": match["signal"],
    }


def compare_stocks(
    signal_records: list[dict],
    moving_avg_records: list[dict],
    symbol1: str,
    symbol2: str,
) -> dict:
    """Side-by-side comparison of two stocks.

    Args:
        signal_records: Market signal dicts (symbol, sector, latest_close, rsi,
            macd_histogram, signal).
        moving_avg_records: Moving average dicts (symbol, sma_5, sma_20, vwap_20).
        symbol1: First stock ticker.
        symbol2: Second stock ticker.

    Returns:
        Dict with {"comparison": {sym1: {...}, sym2: {...}}}, or {"error": "..."}.
    """
    symbol1 = symbol1.upper().strip()
    symbol2 = symbol2.upper().strip()

    avg_by_symbol = {r["symbol"]: r for r in moving_avg_records}

    stocks = {}
    for sym in (symbol1, symbol2):
        sig = next((r for r in signal_records if r["symbol"] == sym), None)
        if not sig:
            return {"error": f"Could not find data for {sym}"}

        avg = avg_by_symbol.get(sym, {})
        stocks[sym] = {
            "sector": sig["sector"],
            "price": round(sig["latest_close"], 2),
            "rsi": round(sig["rsi"], 2) if sig.get("rsi") is not None else None,
            "macd_histogram": round(sig["macd_histogram"], 4) if sig.get("macd_histogram") is not None else None,
            "signal": sig["signal"],
            "sma_5": round(avg["sma_5"], 2) if avg.get("sma_5") is not None else None,
            "sma_20": round(avg["sma_20"], 2) if avg.get("sma_20") is not None else None,
            "vwap_20": round(avg["vwap_20"], 2) if avg.get("vwap_20") is not None else None,
        }

    return {"comparison": stocks}


def get_market_summary(
    daily_records: list[dict],
    signal_records: list[dict],
    spike_records: list[dict],
) -> dict:
    """Overall market snapshot — top gainers, losers, most active, signals.

    Args:
        daily_records: Daily summary dicts (symbol, avg_change_pct, total_volume).
        signal_records: Market signal dicts (symbol, signal).
        spike_records: Volume spike dicts (symbol, volume_ratio, is_spike).

    Returns:
        Dict with market snapshot, or {"error": "..."} if empty.
    """
    if not daily_records:
        return {"error": "No daily data available"}

    # Sort by change for gainers/losers
    sorted_by_change = sorted(daily_records, key=lambda r: r["avg_change_pct"], reverse=True)
    top_gainers = [{"symbol": r["symbol"], "change_pct": round(r["avg_change_pct"], 2)} for r in sorted_by_change[:3]]
    top_losers = [
        {"symbol": r["symbol"], "change_pct": round(r["avg_change_pct"], 2)} for r in reversed(sorted_by_change[-3:])
    ]

    # Most active by volume
    sorted_by_volume = sorted(daily_records, key=lambda r: r["total_volume"], reverse=True)
    most_active = [{"symbol": r["symbol"], "volume": r["total_volume"]} for r in sorted_by_volume[:3]]

    # Signal distribution
    signal_counts: dict[str, int] = {}
    for r in signal_records:
        sig = r["signal"]
        signal_counts[sig] = signal_counts.get(sig, 0) + 1

    # Volume spikes
    spikes = [{"symbol": r["symbol"], "ratio": round(r["volume_ratio"], 2)} for r in spike_records if r.get("is_spike")]

    return {
        "date": str(daily_records[0]["date"]),
        "total_stocks": len(daily_records),
        "top_gainers": top_gainers,
        "top_losers": top_losers,
        "most_active": most_active,
        "signal_distribution": signal_counts,
        "volume_spikes_today": spikes,
    }
