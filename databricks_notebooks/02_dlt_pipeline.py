# Databricks notebook source
# MAGIC %md
# MAGIC # MarketMind AI — DLT Pipeline
# MAGIC Bronze → Silver → Gold medallion architecture.
# MAGIC
# MAGIC **Run this as a Delta Live Tables pipeline, NOT as a regular notebook.**
# MAGIC
# MAGIC To set up:
# MAGIC 1. Go to Pipelines → Create Pipeline
# MAGIC 2. Set this notebook as the source
# MAGIC 3. Set target schema: "marketmind"
# MAGIC 4. Set storage location: "/mnt/marketmind/dlt/"
# MAGIC 5. Click Start

# COMMAND ----------

import dlt
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    LongType,
)

# COMMAND ----------

# ── Configuration ──────────────────────────────────────

RAW_PRICES_PATH = "/mnt/marketmind/raw/prices/"

SYMBOL_SECTORS = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "GOOGL": "Technology",
    "AMZN": "Technology",
    "JPM": "Finance",
    "GS": "Finance",
    "BAC": "Finance",
    "XOM": "Energy",
    "CVX": "Energy",
    "PFE": "Healthcare",
    "JNJ": "Healthcare",
    "TSLA": "Consumer",
    "WMT": "Consumer",
    "KO": "Consumer",
}

STOCK_PRICE_SCHEMA = StructType([
    StructField("symbol", StringType(), nullable=False),
    StructField("price", DoubleType(), nullable=True),
    StructField("open", DoubleType(), nullable=True),
    StructField("high", DoubleType(), nullable=True),
    StructField("low", DoubleType(), nullable=True),
    StructField("prev_close", DoubleType(), nullable=True),
    StructField("volume", LongType(), nullable=True),
    StructField("market_cap", LongType(), nullable=True),
    StructField("timestamp", StringType(), nullable=True),
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze Layer — Raw Data Landing

# COMMAND ----------

@dlt.table(
    name="stock_prices_bronze",
    comment="Raw stock prices from Yahoo Finance — no transformations",
    table_properties={
        "quality": "bronze",
        "pipelines.autoOptimize.managed": "true",
    },
)
def stock_prices_bronze():
    return (
        spark.readStream
        .format("json")
        .schema(STOCK_PRICE_SCHEMA)
        .option("maxFilesPerTrigger", 1)
        .load(RAW_PRICES_PATH)
        .withColumn("ingest_time", F.current_timestamp())
        .withColumn("trade_time", F.to_timestamp(F.col("timestamp")))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver Layer — Clean, Validate, Enrich

# COMMAND ----------

# Enriched view — single source for both Silver and Quarantine
@dlt.view(
    name="stock_prices_enriched_v",
    comment="Enriched view with derived fields — feeds Silver and Quarantine",
)
def stock_prices_enriched_v():
    # Build sector mapping expression
    sector_mapping = F.create_map(*[
        item for pair in SYMBOL_SECTORS.items() for item in (F.lit(pair[0]), F.lit(pair[1]))
    ])

    return (
        dlt.read_stream("stock_prices_bronze")
        .withColumn("price_change", F.col("price") - F.col("prev_close"))
        .withColumn("price_change_pct",
            (F.col("price") - F.col("prev_close")) / F.col("prev_close") * 100
        )
        .withColumn("sector",
            F.coalesce(sector_mapping[F.col("symbol")], F.lit("Unknown"))
        )
        .withColumn("volume_category",
            F.when(F.col("volume") >= 75_000_000, "very_high")
             .when(F.col("volume") >= 25_000_000, "high")
             .when(F.col("volume") >= 10_000_000, "medium")
             .otherwise("low")
        )
    )

# COMMAND ----------

@dlt.table(
    name="stock_prices_silver",
    comment="Cleaned and enriched stock prices — quality validated",
    table_properties={
        "quality": "silver",
        "pipelines.autoOptimize.managed": "true",
    },
)
@dlt.expect_or_drop("valid_price", "price > 0")
@dlt.expect_or_drop("valid_symbol", "symbol IS NOT NULL")
@dlt.expect_or_drop("valid_volume", "volume >= 0")
@dlt.expect_or_drop("valid_timestamp", "timestamp IS NOT NULL")
def stock_prices_silver():
    return dlt.read_stream("stock_prices_enriched_v")

# COMMAND ----------

@dlt.table(
    name="stock_prices_quarantine",
    comment="Rejected records that failed quality checks",
    table_properties={
        "quality": "quarantine",
    },
)
def stock_prices_quarantine():
    return (
        dlt.read_stream("stock_prices_enriched_v")
        .filter(
            (F.col("price").isNull()) | (F.col("price") <= 0) |
            (F.col("symbol").isNull()) |
            (F.col("volume").isNull()) | (F.col("volume") < 0) |
            (F.col("timestamp").isNull())
        )
        .withColumn("quarantine_reason",
            F.when(F.col("price").isNull() | (F.col("price") <= 0), "Invalid price")
             .when(F.col("symbol").isNull(), "Invalid symbol")
             .when(F.col("volume").isNull() | (F.col("volume") < 0), "Invalid volume")
             .when(F.col("timestamp").isNull(), "Invalid timestamp")
             .otherwise("Unknown")
        )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold Layer — Analytics-Ready Aggregations

# COMMAND ----------

@dlt.table(
    name="stock_daily_summary",
    comment="Daily OHLCV summary per stock — partitioned by date",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true",
    },
    partition_cols=["date"],
)
def stock_daily_summary():
    return (
        dlt.read("stock_prices_silver")
        .withColumn("date", F.to_date("trade_time"))
        .groupBy("symbol", "date", "sector")
        .agg(
            F.first("open").alias("day_open"),
            F.max("high").alias("day_high"),
            F.min("low").alias("day_low"),
            F.last("price").alias("day_close"),
            F.sum("volume").alias("total_volume"),
            F.count("*").alias("trade_count"),
            F.avg("price_change_pct").alias("avg_change_pct"),
        )
    )

# COMMAND ----------

@dlt.table(
    name="sector_performance",
    comment="Aggregated sector performance metrics",
    table_properties={
        "quality": "gold",
    },
)
def sector_performance():
    return (
        dlt.read("stock_prices_silver")
        .withColumn("date", F.to_date("trade_time"))
        .groupBy("sector", "date")
        .agg(
            F.avg("price_change_pct").alias("avg_change_pct"),
            F.sum("volume").alias("total_volume"),
            F.count("*").alias("stock_count"),
        )
    )
