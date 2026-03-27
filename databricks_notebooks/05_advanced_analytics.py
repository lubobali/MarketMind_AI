# Databricks notebook source
# MAGIC %md
# MAGIC # MarketMind AI — Phase 5: Advanced Spark Analytics
# MAGIC
# MAGIC **Module 3 coverage**: Custom UDFs, Window Functions, Optimization
# MAGIC
# MAGIC **Reads from**:
# MAGIC - `bootcamp_students.lubo_marketmind_ai.stock_prices_silver`
# MAGIC - `bootcamp_students.lubo_marketmind_ai.stock_daily_summary`
# MAGIC
# MAGIC **Writes**:
# MAGIC - `bootcamp_students.lubo_marketmind_ai.technical_indicators` — RSI, MACD, Bollinger per symbol
# MAGIC - `bootcamp_students.lubo_marketmind_ai.market_signals` — buy/sell/hold classification
# MAGIC - `bootcamp_students.lubo_marketmind_ai.moving_averages` — 5/20/50-day MAs + VWAP
# MAGIC - `bootcamp_students.lubo_marketmind_ai.sector_rankings` — daily rank by performance within sector
# MAGIC - `bootcamp_students.lubo_marketmind_ai.volume_spikes` — anomalous volume days
# MAGIC
# MAGIC Run this as a **regular notebook** on your cluster (not DLT).

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from pyspark.sql.window import Window

# COMMAND ----------

# ── Configuration ────────────────────────────────────────────
CATALOG = "bootcamp_students"
SCHEMA = "lubo_marketmind_ai"

def table(name: str) -> str:
    """Full 3-level table name."""
    return f"{CATALOG}.{SCHEMA}.{name}"

# Source tables
SILVER_TABLE = table("stock_prices_silver")
DAILY_SUMMARY_TABLE = table("stock_daily_summary")

# Target tables
TECHNICAL_INDICATORS_TABLE = table("technical_indicators")
MARKET_SIGNALS_TABLE = table("market_signals")
MOVING_AVERAGES_TABLE = table("moving_averages")
SECTOR_RANKINGS_TABLE = table("sector_rankings")
VOLUME_SPIKES_TABLE = table("volume_spikes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Custom UDFs — Technical Indicators
# MAGIC
# MAGIC Register Python UDFs for RSI, MACD, Bollinger Bands, and signal classification.
# MAGIC These operate on arrays of closing prices collected per symbol.

# COMMAND ----------

# ── RSI UDF ──────────────────────────────────────────────────
# Relative Strength Index using Wilder's smoothing (period=14)

def _calculate_rsi(prices, period=14):
    """Calculate RSI from a list of closing prices."""
    if not prices or len(prices) < period + 1:
        return None

    changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains = [max(0.0, c) for c in changes[:period]]
    losses = [max(0.0, -c) for c in changes[:period]]

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    for c in changes[period:]:
        avg_gain = (avg_gain * (period - 1) + max(0.0, c)) / period
        avg_loss = (avg_loss * (period - 1) + max(0.0, -c)) / period

    if avg_gain == 0.0 and avg_loss == 0.0:
        return 50.0
    if avg_loss == 0.0:
        return 100.0

    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 4)

calculate_rsi_udf = udf(_calculate_rsi, DoubleType())

# COMMAND ----------

# ── EMA helper (used by MACD) ────────────────────────────────

def _ema(values, period):
    """Exponential Moving Average series."""
    if len(values) < period:
        return []
    k = 2.0 / (period + 1)
    sma = sum(values[:period]) / period
    result = [sma]
    for v in values[period:]:
        result.append(v * k + result[-1] * (1 - k))
    return result

# COMMAND ----------

# ── MACD UDF ─────────────────────────────────────────────────
# MACD(12,26,9): returns struct with macd_line, signal_line, histogram

MACD_SCHEMA = StructType([
    StructField("macd_line", DoubleType()),
    StructField("signal_line", DoubleType()),
    StructField("histogram", DoubleType()),
])

def _calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD from a list of closing prices."""
    if not prices or len(prices) < slow:
        return None

    fast_ema = _ema(prices, fast)
    slow_ema = _ema(prices, slow)

    offset = len(fast_ema) - len(slow_ema)
    fast_aligned = fast_ema[offset:]

    macd_series = [f - s for f, s in zip(fast_aligned, slow_ema)]

    if len(macd_series) < signal:
        return (round(macd_series[-1], 4), round(macd_series[-1], 4), 0.0)

    signal_ema = _ema(macd_series, signal)
    macd_val = round(macd_series[-1], 4)
    signal_val = round(signal_ema[-1], 4)
    hist_val = round(macd_val - signal_val, 4)

    return (macd_val, signal_val, hist_val)

calculate_macd_udf = udf(_calculate_macd, MACD_SCHEMA)

# COMMAND ----------

# ── Bollinger Bands UDF ──────────────────────────────────────
# Returns struct with upper, middle, lower (period=20, 2 std devs)

BOLLINGER_SCHEMA = StructType([
    StructField("upper", DoubleType()),
    StructField("middle", DoubleType()),
    StructField("lower", DoubleType()),
])

def _calculate_bollinger(prices, period=20, num_std=2.0):
    """Calculate Bollinger Bands from a list of closing prices."""
    if not prices or len(prices) < period:
        return None

    window = prices[-period:]
    middle = sum(window) / period
    variance = sum((p - middle) ** 2 for p in window) / period
    std = variance ** 0.5
    band_offset = num_std * std

    return (
        round(middle + band_offset, 4),
        round(middle, 4),
        round(middle - band_offset, 4),
    )

calculate_bollinger_udf = udf(_calculate_bollinger, BOLLINGER_SCHEMA)

# COMMAND ----------

# ── Signal Classifier UDF ────────────────────────────────────
# Combines RSI + MACD + Bollinger → strong_buy/buy/hold/sell/strong_sell

def _classify_signal(rsi, macd_histogram, price, bb_upper, bb_lower):
    """Classify trading signal from technical indicators."""
    if any(v is None for v in (rsi, macd_histogram, price, bb_upper, bb_lower)):
        return "hold"

    score = 0

    # RSI component
    if rsi < 30:
        score += 2
    elif rsi < 45:
        score += 1
    elif rsi > 70:
        score -= 2
    elif rsi > 55:
        score -= 1

    # MACD histogram component
    if macd_histogram > 0:
        score += 1
    elif macd_histogram < 0:
        score -= 1

    # Bollinger Band position
    bb_range = bb_upper - bb_lower
    if bb_range > 0:
        bb_position = (price - bb_lower) / bb_range
        if bb_position < 0.15:
            score += 1
        elif bb_position > 0.85:
            score -= 1

    if score >= 3:
        return "strong_buy"
    elif score >= 1:
        return "buy"
    elif score <= -3:
        return "strong_sell"
    elif score <= -1:
        return "sell"
    return "hold"

classify_signal_udf = udf(_classify_signal, StringType())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute Technical Indicators Per Symbol
# MAGIC
# MAGIC Collect closing prices per symbol (ordered by date), then apply UDFs.

# COMMAND ----------

# Read daily summary — one row per symbol per date
daily_df = spark.table(DAILY_SUMMARY_TABLE)

print(f"Daily summary rows: {daily_df.count()}")
daily_df.printSchema()

# COMMAND ----------

# Collect ordered closing prices per symbol as an array
prices_per_symbol = (
    daily_df
    .orderBy("symbol", "date")
    .groupBy("symbol")
    .agg(
        F.collect_list("day_close").alias("prices"),
        F.last("day_close").alias("latest_close"),
        F.last("date").alias("latest_date"),
        F.last("sector").alias("sector"),
    )
)

print(f"Symbols with price history: {prices_per_symbol.count()}")
prices_per_symbol.select("symbol", F.size("prices").alias("num_days")).show()

# COMMAND ----------

# Apply all technical indicator UDFs
indicators_df = (
    prices_per_symbol
    .withColumn("rsi", calculate_rsi_udf(F.col("prices")))
    .withColumn("macd", calculate_macd_udf(F.col("prices")))
    .withColumn("bollinger", calculate_bollinger_udf(F.col("prices")))
    .select(
        "symbol",
        "sector",
        "latest_date",
        "latest_close",
        "rsi",
        F.col("macd.macd_line").alias("macd_line"),
        F.col("macd.signal_line").alias("macd_signal"),
        F.col("macd.histogram").alias("macd_histogram"),
        F.col("bollinger.upper").alias("bb_upper"),
        F.col("bollinger.middle").alias("bb_middle"),
        F.col("bollinger.lower").alias("bb_lower"),
    )
)

indicators_df.show(truncate=False)

# COMMAND ----------

# Write technical indicators table
(
    indicators_df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TECHNICAL_INDICATORS_TABLE)
)

print(f"Wrote {indicators_df.count()} rows to {TECHNICAL_INDICATORS_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Market Signals — Buy/Sell/Hold Classification

# COMMAND ----------

signals_df = (
    indicators_df
    .withColumn(
        "signal",
        classify_signal_udf(
            F.col("rsi"),
            F.col("macd_histogram"),
            F.col("latest_close"),
            F.col("bb_upper"),
            F.col("bb_lower"),
        ),
    )
    .select(
        "symbol", "sector", "latest_date", "latest_close",
        "rsi", "macd_line", "macd_signal", "macd_histogram",
        "bb_upper", "bb_middle", "bb_lower", "signal",
    )
)

signals_df.show(truncate=False)

# COMMAND ----------

# Write market signals table
(
    signals_df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(MARKET_SIGNALS_TABLE)
)

print(f"Wrote {signals_df.count()} rows to {MARKET_SIGNALS_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Window Functions
# MAGIC
# MAGIC - Moving averages (5/20/50-day SMA)
# MAGIC - VWAP (Volume-Weighted Average Price)
# MAGIC - Sector ranking by daily performance
# MAGIC - Volume spike detection

# COMMAND ----------

# MAGIC %md
# MAGIC ### Moving Averages + VWAP

# COMMAND ----------

# Window definitions
w5 = Window.partitionBy("symbol").orderBy("date").rowsBetween(-4, 0)
w20 = Window.partitionBy("symbol").orderBy("date").rowsBetween(-19, 0)
w50 = Window.partitionBy("symbol").orderBy("date").rowsBetween(-49, 0)

# Count window — to know if we have enough data for each MA
w5_count = Window.partitionBy("symbol").orderBy("date").rowsBetween(-4, 0)
w20_count = Window.partitionBy("symbol").orderBy("date").rowsBetween(-19, 0)
w50_count = Window.partitionBy("symbol").orderBy("date").rowsBetween(-49, 0)

ma_df = (
    daily_df
    .withColumn("sma_5", F.avg("day_close").over(w5))
    .withColumn("sma_20", F.avg("day_close").over(w20))
    .withColumn("sma_50", F.avg("day_close").over(w50))
    # Null out MAs where we don't have a full window
    .withColumn("row_num", F.row_number().over(
        Window.partitionBy("symbol").orderBy("date")
    ))
    .withColumn("sma_5",
        F.when(F.col("row_num") >= 5, F.col("sma_5"))
    )
    .withColumn("sma_20",
        F.when(F.col("row_num") >= 20, F.col("sma_20"))
    )
    .withColumn("sma_50",
        F.when(F.col("row_num") >= 50, F.col("sma_50"))
    )
    # VWAP = cumulative(price * volume) / cumulative(volume) per symbol per date
    # For daily data, VWAP per row = day_close (single price per day).
    # We compute a rolling VWAP over the 20-day window instead.
    .withColumn("price_x_volume", F.col("day_close") * F.col("total_volume"))
    .withColumn(
        "vwap_20",
        F.sum("price_x_volume").over(w20) / F.sum("total_volume").over(w20),
    )
    .withColumn("vwap_20",
        F.when(F.col("row_num") >= 20, F.col("vwap_20"))
    )
    .select(
        "symbol", "sector", "date", "day_close", "total_volume",
        F.round("sma_5", 4).alias("sma_5"),
        F.round("sma_20", 4).alias("sma_20"),
        F.round("sma_50", 4).alias("sma_50"),
        F.round("vwap_20", 4).alias("vwap_20"),
    )
)

ma_df.filter(F.col("symbol") == "NVDA").orderBy(F.desc("date")).show(10, truncate=False)

# COMMAND ----------

# Write moving averages table — partitioned by date for efficient queries
(
    ma_df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("date")
    .saveAsTable(MOVING_AVERAGES_TABLE)
)

print(f"Wrote {ma_df.count()} rows to {MOVING_AVERAGES_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sector Rankings — Rank stocks by daily change within sector

# COMMAND ----------

sector_rank_window = Window.partitionBy("sector", "date").orderBy(F.desc("avg_change_pct"))

rankings_df = (
    daily_df
    .withColumn("sector_rank", F.rank().over(sector_rank_window))
    .withColumn("sector_size", F.count("*").over(Window.partitionBy("sector", "date")))
    .withColumn("percentile",
        F.round((1 - (F.col("sector_rank") - 1) / F.col("sector_size")) * 100, 1)
    )
    .select(
        "symbol", "sector", "date", "avg_change_pct",
        "sector_rank", "sector_size", "percentile",
    )
)

# Show today's rankings
rankings_df.orderBy("date", "sector", "sector_rank").show(30, truncate=False)

# COMMAND ----------

# Write sector rankings table
(
    rankings_df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("date")
    .saveAsTable(SECTOR_RANKINGS_TABLE)
)

print(f"Wrote {rankings_df.count()} rows to {SECTOR_RANKINGS_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Volume Spike Detection
# MAGIC
# MAGIC A volume spike = current day's volume > 2x the 20-day rolling average.

# COMMAND ----------

vol_window = Window.partitionBy("symbol").orderBy("date").rowsBetween(-20, -1)

spikes_df = (
    daily_df
    .withColumn("avg_volume_20d", F.avg("total_volume").over(vol_window))
    .withColumn("volume_ratio",
        F.when(F.col("avg_volume_20d") > 0,
            F.col("total_volume") / F.col("avg_volume_20d")
        ).otherwise(None)
    )
    .withColumn("is_spike", F.col("volume_ratio") > 2.0)
    .filter(F.col("avg_volume_20d").isNotNull())  # skip first 20 days
    .select(
        "symbol", "sector", "date", "total_volume",
        F.round("avg_volume_20d", 0).cast("long").alias("avg_volume_20d"),
        F.round("volume_ratio", 2).alias("volume_ratio"),
        "is_spike",
    )
)

# Show only spikes
spike_count = spikes_df.filter(F.col("is_spike")).count()
print(f"Volume spikes detected: {spike_count}")
spikes_df.filter(F.col("is_spike")).orderBy(F.desc("volume_ratio")).show(20, truncate=False)

# COMMAND ----------

# Write volume spikes table
(
    spikes_df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("date")
    .saveAsTable(VOLUME_SPIKES_TABLE)
)

print(f"Wrote {spikes_df.count()} rows to {VOLUME_SPIKES_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3: Optimization
# MAGIC
# MAGIC Z-order key tables by symbol for fast lookups. Run OPTIMIZE + VACUUM.

# COMMAND ----------

# Z-order the new tables by symbol
for tbl in [
    TECHNICAL_INDICATORS_TABLE,
    MARKET_SIGNALS_TABLE,
    MOVING_AVERAGES_TABLE,
    SECTOR_RANKINGS_TABLE,
    VOLUME_SPIKES_TABLE,
]:
    print(f"Optimizing {tbl}...")
    spark.sql(f"OPTIMIZE {tbl} ZORDER BY (symbol)")
    print(f"  Z-ordered by symbol")

print("All tables optimized.")

# COMMAND ----------

# Vacuum old files (retain 168 hours = 7 days by default)
for tbl in [
    TECHNICAL_INDICATORS_TABLE,
    MARKET_SIGNALS_TABLE,
    MOVING_AVERAGES_TABLE,
    SECTOR_RANKINGS_TABLE,
    VOLUME_SPIKES_TABLE,
]:
    spark.sql(f"VACUUM {tbl}")
    print(f"Vacuumed {tbl}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary — Tables Created
# MAGIC
# MAGIC | Table | Description | Rows |
# MAGIC |-------|-------------|------|
# MAGIC | `technical_indicators` | RSI, MACD, Bollinger per symbol | 15 |
# MAGIC | `market_signals` | Buy/sell/hold classification | 15 |
# MAGIC | `moving_averages` | 5/20/50-day SMA + VWAP | ~15 x days |
# MAGIC | `sector_rankings` | Daily rank within sector | ~15 x days |
# MAGIC | `volume_spikes` | Volume anomaly detection | ~15 x days |

# COMMAND ----------

# Final verification — show all new tables
for tbl_name in [
    "technical_indicators",
    "market_signals",
    "moving_averages",
    "sector_rankings",
    "volume_spikes",
]:
    full_name = table(tbl_name)
    count = spark.table(full_name).count()
    print(f"{full_name}: {count} rows")
