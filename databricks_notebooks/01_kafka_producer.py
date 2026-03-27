# Databricks notebook source
# MAGIC %md
# MAGIC # MarketMind AI — Stock Price Producer
# MAGIC Fetches real-time stock prices from Yahoo Finance and writes them
# MAGIC as JSON files for Spark Structured Streaming to consume.

# COMMAND ----------

# MAGIC %pip install yfinance
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import json
import os
from datetime import datetime, timezone

import yfinance as yf

# COMMAND ----------

# ── Configuration ──────────────────────────────────────

TRACKED_SYMBOLS = [
    # Technology
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    # Finance
    "JPM", "GS", "BAC",
    # Energy
    "XOM", "CVX",
    # Healthcare
    "PFE", "JNJ",
    # Consumer
    "TSLA", "WMT", "KO",
]

# Where to write JSON files — Spark readStream will consume from here
RAW_PRICES_PATH = "/mnt/marketmind/raw/prices/"

# COMMAND ----------

# ── Create output directory ────────────────────────────

dbutils.fs.mkdirs(RAW_PRICES_PATH)
print(f"Output directory ready: {RAW_PRICES_PATH}")

# COMMAND ----------

def fetch_stock_prices(symbols):
    """Fetch current prices using yf.Tickers (batch — one HTTP call for all symbols)."""
    records = []
    now = datetime.now(timezone.utc).isoformat()

    # Batch fetch — reduces latency vs per-symbol yf.Ticker calls
    tickers = yf.Tickers(" ".join(symbols))

    for symbol in symbols:
        try:
            info = tickers.tickers[symbol].info

            price = info.get("regularMarketPrice")
            if price is None:
                print(f"  ⚠️ No price data for {symbol} — skipping")
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
            print(f"  ✅ {symbol}: ${price:.2f}")

        except Exception as e:
            print(f"  ❌ {symbol}: {e}")

    return records

# COMMAND ----------

def write_to_dbfs(records, output_dir):
    """Write records as newline-delimited JSON to DBFS for Spark readStream."""
    filename = f"prices_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    filepath = f"{output_dir}{filename}"

    # Build newline-delimited JSON
    content = "\n".join(json.dumps(r) for r in records)

    # Write to DBFS
    dbutils.fs.put(filepath, content, overwrite=True)
    print(f"📁 Wrote {len(records)} records → {filepath}")
    return filepath

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the Producer
# MAGIC Fetch prices for all tracked symbols and write to DBFS.

# COMMAND ----------

print(f"🚀 Fetching prices for {len(TRACKED_SYMBOLS)} symbols...")
prices = fetch_stock_prices(TRACKED_SYMBOLS)
print(f"\n📊 Got {len(prices)} prices")

# COMMAND ----------

if prices:
    path = write_to_dbfs(prices, RAW_PRICES_PATH)
    print(f"\n✅ Data written. Ready for Bronze layer.")
else:
    print("❌ No prices fetched — check yfinance connectivity")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify the data

# COMMAND ----------

# List files in output directory
display(dbutils.fs.ls(RAW_PRICES_PATH))

# COMMAND ----------

# Read the JSON file with Spark to verify format
df = spark.read.json(RAW_PRICES_PATH)
display(df)
print(f"Schema: {df.schema.simpleString()}")
print(f"Record count: {df.count()}")
