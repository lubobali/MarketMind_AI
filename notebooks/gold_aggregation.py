# Databricks notebook source
# MAGIC %md
# MAGIC # MarketMind AI — Gold Layer
# MAGIC Analytics-ready aggregations built from Silver data.
# MAGIC
# MAGIC **Tables:**
# MAGIC - stock_daily_summary: OHLCV per symbol per day
# MAGIC - sector_performance: Aggregated sector metrics
# MAGIC
# MAGIC **Optimization:**
# MAGIC - Partitioned by date
# MAGIC - Z-ordered by symbol

# import dlt
# from pyspark.sql import functions as F
# from pyspark.sql.window import Window

import logging

logger = logging.getLogger(__name__)

# ── DLT table names ────────────────────────────────────
DAILY_SUMMARY_TABLE = "stock_daily_summary"
SECTOR_PERFORMANCE_TABLE = "sector_performance"


# ── Daily summary ──────────────────────────────────────
def compute_daily_summary(records: list[dict], symbol: str, date: str) -> dict:
    """Aggregate intraday records into a daily OHLCV summary.

    Args:
        records: List of intraday price records for one symbol.
        symbol: The stock ticker.
        date: The trading date (YYYY-MM-DD).

    Returns:
        Daily summary with OHLCV + trade_count.
    """
    sorted_records = sorted(records, key=lambda r: r["timestamp"])

    return {
        "symbol": symbol,
        "date": date,
        "day_open": sorted_records[0]["open"],
        "day_high": max(r["high"] for r in sorted_records),
        "day_low": min(r["low"] for r in sorted_records),
        "day_close": sorted_records[-1]["price"],
        "total_volume": sum(r["volume"] for r in sorted_records),
        "trade_count": len(sorted_records),
    }


# ── Sector performance ────────────────────────────────
def compute_sector_performance(records: list[dict], sector: str) -> dict:
    """Aggregate performance metrics for a sector.

    Args:
        records: List of enriched Silver records for one sector.
        sector: The sector name.

    Returns:
        Sector summary with avg change, volume, top gainer/loser.
    """
    changes = [r["price_change_pct"] for r in records]
    volumes = [r["volume"] for r in records]

    sorted_by_change = sorted(records, key=lambda r: r["price_change_pct"])

    return {
        "sector": sector,
        "avg_change_pct": sum(changes) / len(changes),
        "total_volume": sum(volumes),
        "stock_count": len(records),
        "top_gainer": sorted_by_change[-1]["symbol"],
        "top_loser": sorted_by_change[0]["symbol"],
    }


# ── DLT table definitions (runs in Databricks only) ────
# Uncomment when running as a DLT pipeline:
#
# @dlt.table(
#     name=DAILY_SUMMARY_TABLE,
#     comment="Daily OHLCV summary per stock — partitioned by date, Z-ordered by symbol",
#     table_properties={"quality": "gold", "pipelines.autoOptimize.managed": "true"},
#     partition_cols=["date"],
# )
# def stock_daily_summary():
#     w = Window.partitionBy("symbol", "date").orderBy("trade_time")
#     w_desc = Window.partitionBy("symbol", "date").orderBy(F.desc("trade_time"))
#
#     return (
#         dlt.read(SILVER_TABLE_NAME)
#         .withColumn("date", F.to_date("trade_time"))
#         .withColumn("row_asc", F.row_number().over(w))
#         .withColumn("row_desc", F.row_number().over(w_desc))
#         .groupBy("symbol", "date")
#         .agg(
#             F.first(F.when(F.col("row_asc") == 1, F.col("open"))).alias("day_open"),
#             F.max("high").alias("day_high"),
#             F.min("low").alias("day_low"),
#             F.first(F.when(F.col("row_desc") == 1, F.col("price"))).alias("day_close"),
#             F.sum("volume").alias("total_volume"),
#             F.count("*").alias("trade_count"),
#         )
#     )
#
# @dlt.table(
#     name=SECTOR_PERFORMANCE_TABLE,
#     comment="Aggregated sector performance metrics",
#     table_properties={"quality": "gold"},
# )
# def sector_performance():
#     return (
#         dlt.read(SILVER_TABLE_NAME)
#         .withColumn("date", F.to_date("trade_time"))
#         .groupBy("sector", "date")
#         .agg(
#             F.avg("price_change_pct").alias("avg_change_pct"),
#             F.sum("volume").alias("total_volume"),
#             F.count("*").alias("stock_count"),
#         )
#     )
