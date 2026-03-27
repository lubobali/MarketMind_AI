# Databricks notebook source
# MAGIC %md
# MAGIC # MarketMind AI — Stock Price Producer
# MAGIC Fetches real-time stock prices from Yahoo Finance and writes them
# MAGIC as JSON files for Spark Structured Streaming to consume.
# MAGIC
# MAGIC Can also publish to Kafka topic if configured.

import json
import logging
import os
from datetime import datetime, timezone

import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_stock_prices(symbols: list[str]) -> list[dict]:
    """Fetch current prices for a list of stock symbols.

    Uses yf.Tickers for batch fetching — one HTTP call for all symbols
    instead of one per symbol, reducing latency significantly.

    Args:
        symbols: List of ticker symbols (e.g., ["AAPL", "NVDA"])

    Returns:
        List of price records. Failed symbols are skipped.
    """
    records = []
    now = datetime.now(timezone.utc).isoformat()

    # Batch fetch — single request for all symbols
    try:
        tickers = yf.Tickers(" ".join(symbols))
    except Exception as e:
        logger.warning("Failed to create batch ticker request: %s", e)
        return records

    for symbol in symbols:
        try:
            info = tickers.tickers[symbol].info

            price = info.get("regularMarketPrice")
            if price is None:
                logger.warning("No price data for %s — skipping (invalid or delisted symbol)", symbol)
                continue

            record = {
                "symbol": symbol,
                "price": price,
                "open": info.get("regularMarketOpen"),
                "high": info.get("regularMarketDayHigh"),
                "low": info.get("regularMarketDayLow"),
                "prev_close": info.get("regularMarketPreviousClose"),
                "volume": info.get("regularMarketVolume"),
                "market_cap": info.get("marketCap"),
                "timestamp": now,
            }
            records.append(record)

        except Exception as e:
            logger.warning("Failed to fetch %s: %s", symbol, e)

    return records


def write_to_json_files(records: list[dict], output_dir: str) -> str:
    """Write records as a newline-delimited JSON file for Spark readStream.

    Args:
        records: List of stock price records.
        output_dir: Directory to write JSON files into.

    Returns:
        Path to the written file.
    """
    os.makedirs(output_dir, exist_ok=True)

    filename = f"prices_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    logger.info("Wrote %d records to %s", len(records), filepath)
    return filepath


# ── Main loop (runs in Databricks notebook) ─────────────
if __name__ == "__main__":
    import time

    from config.settings import PRICE_POLL_INTERVAL, RAW_PRICES_PATH, TRACKED_SYMBOLS

    print(f"MarketMind AI — Streaming {len(TRACKED_SYMBOLS)} symbols every {PRICE_POLL_INTERVAL}s")

    while True:
        prices = fetch_stock_prices(TRACKED_SYMBOLS)
        if prices:
            path = write_to_json_files(prices, RAW_PRICES_PATH)
            print(f"[{datetime.now(timezone.utc).isoformat()}] Wrote {len(prices)} prices → {path}")
        else:
            print(f"[{datetime.now(timezone.utc).isoformat()}] No prices fetched — retrying")

        time.sleep(PRICE_POLL_INTERVAL)
