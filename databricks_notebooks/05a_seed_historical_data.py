# Databricks notebook source
# MAGIC %md
# MAGIC # MarketMind AI — Seed Historical Data
# MAGIC
# MAGIC Fetches 60 days of daily OHLCV from Yahoo Finance and writes to
# MAGIC `stock_daily_summary` so technical indicators can compute.
# MAGIC
# MAGIC **Run this ONCE**, then re-run `05_advanced_analytics.py`.

# COMMAND ----------

# MAGIC %pip install yfinance
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import yfinance as yf
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    DateType,
    IntegerType,
)
from datetime import datetime, timedelta

# COMMAND ----------

# ── Configuration ────────────────────────────────────────────
CATALOG = "bootcamp_students"
SCHEMA = "lubo_marketmind_ai"
TABLE = f"{CATALOG}.{SCHEMA}.stock_daily_summary"

SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "JPM", "GS", "BAC",
    "XOM", "CVX",
    "PFE", "JNJ",
    "TSLA", "WMT", "KO",
]

SYMBOL_SECTORS = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOGL": "Technology", "AMZN": "Technology",
    "JPM": "Finance", "GS": "Finance", "BAC": "Finance",
    "XOM": "Energy", "CVX": "Energy",
    "PFE": "Healthcare", "JNJ": "Healthcare",
    "TSLA": "Consumer", "WMT": "Consumer", "KO": "Consumer",
}

# COMMAND ----------

# ── Fetch 60 days of daily data from Yahoo Finance ───────────
end_date = datetime.now()
start_date = end_date - timedelta(days=90)  # fetch 90 calendar days ≈ 60 trading days

print(f"Fetching {len(SYMBOLS)} symbols from {start_date.date()} to {end_date.date()}...")

all_rows = []

for symbol in SYMBOLS:
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

    if hist.empty:
        print(f"  {symbol}: NO DATA")
        continue

    for date_idx, row in hist.iterrows():
        trade_date = date_idx.date()
        prev_close = row.get("Close", 0)  # simplified — used for change calc

        all_rows.append({
            "symbol": symbol,
            "date": str(trade_date),
            "sector": SYMBOL_SECTORS[symbol],
            "day_open": float(row["Open"]),
            "day_high": float(row["High"]),
            "day_low": float(row["Low"]),
            "day_close": float(row["Close"]),
            "total_volume": int(row["Volume"]),
            "trade_count": 1,
            "avg_change_pct": 0.0,  # will be computed below
        })

    print(f"  {symbol}: {len(hist)} trading days")

print(f"\nTotal rows: {len(all_rows)}")

# COMMAND ----------

# ── Compute avg_change_pct (daily % change) ──────────────────
# Sort by symbol + date, then compute pct change from previous day

from collections import defaultdict

by_symbol = defaultdict(list)
for row in all_rows:
    by_symbol[row["symbol"]].append(row)

for symbol, rows in by_symbol.items():
    rows.sort(key=lambda r: r["date"])
    for i, row in enumerate(rows):
        if i == 0:
            row["avg_change_pct"] = 0.0
        else:
            prev_close = rows[i - 1]["day_close"]
            if prev_close > 0:
                row["avg_change_pct"] = ((row["day_close"] - prev_close) / prev_close) * 100
            else:
                row["avg_change_pct"] = 0.0

print("Computed daily % change for all rows")

# COMMAND ----------

# ── Create Spark DataFrame and write to Delta ────────────────

schema = StructType([
    StructField("symbol", StringType(), nullable=False),
    StructField("date", StringType(), nullable=True),
    StructField("sector", StringType(), nullable=True),
    StructField("day_open", DoubleType(), nullable=True),
    StructField("day_high", DoubleType(), nullable=True),
    StructField("day_low", DoubleType(), nullable=True),
    StructField("day_close", DoubleType(), nullable=True),
    StructField("total_volume", LongType(), nullable=True),
    StructField("trade_count", IntegerType(), nullable=True),
    StructField("avg_change_pct", DoubleType(), nullable=True),
])

df = spark.createDataFrame(all_rows, schema=schema)

# Cast date string to proper date type
df = df.withColumn("date", F.to_date("date"))

print(f"DataFrame rows: {df.count()}")
df.groupBy("symbol").count().orderBy("symbol").show(20)

# COMMAND ----------

# ── Write — overwrite the table with full history ────────────
(
    df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("date")
    .saveAsTable(TABLE)
)

print(f"Wrote {df.count()} rows to {TABLE}")

# COMMAND ----------

# ── Verify ───────────────────────────────────────────────────
verify_df = spark.table(TABLE)
print(f"Table row count: {verify_df.count()}")
print(f"Date range: {verify_df.agg(F.min('date'), F.max('date')).collect()[0]}")
verify_df.groupBy("symbol").agg(F.count("*").alias("days")).orderBy("symbol").show(20)
