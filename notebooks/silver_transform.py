# Databricks notebook source
# MAGIC %md
# MAGIC # MarketMind AI — Silver Layer
# MAGIC Cleans, validates, and enriches Bronze data.
# MAGIC Bad records go to quarantine with a reason.
# MAGIC
# MAGIC **Data quality rules:**
# MAGIC - price > 0 and not null
# MAGIC - volume >= 0 and not null
# MAGIC - symbol is not null
# MAGIC - timestamp is not null

# import dlt
# from pyspark.sql import functions as F

import logging

from config.settings import SYMBOL_SECTORS

logger = logging.getLogger(__name__)

# ── DLT table configuration ────────────────────────────
SILVER_TABLE_NAME = "stock_prices_silver"
QUARANTINE_TABLE_NAME = "stock_prices_quarantine"

# DLT expectations — used as @dlt.expect_or_drop() decorators
DLT_EXPECTATIONS = {
    "valid_price": "price > 0",
    "valid_symbol": "symbol IS NOT NULL",
    "valid_volume": "volume >= 0",
    "valid_timestamp": "timestamp IS NOT NULL",
}


# ── Volume classification ──────────────────────────────
def classify_volume(volume: int) -> str:
    """Classify trading volume into buckets.

    Args:
        volume: Trading volume (number of shares).

    Returns:
        Category: "very_high", "high", "medium", or "low".
    """
    if volume >= 75_000_000:
        return "very_high"
    elif volume >= 25_000_000:
        return "high"
    elif volume >= 10_000_000:
        return "medium"
    else:
        return "low"


# ── Data quality validation ─────────────────────────────
def validate_record(record: dict) -> tuple[bool, str | None]:
    """Validate a record against Silver quality rules.

    Args:
        record: Stock price record to validate.

    Returns:
        Tuple of (is_valid, reason). reason is None if valid.
    """
    price = record.get("price")
    if price is None:
        return False, "Invalid price: null"
    if price <= 0:
        return False, f"Invalid price: {price} (must be > 0)"

    symbol = record.get("symbol")
    if symbol is None:
        return False, "Invalid symbol: null"

    volume = record.get("volume")
    if volume is None:
        return False, "Invalid volume: null"
    if volume < 0:
        return False, f"Invalid volume: {volume} (must be >= 0)"

    timestamp = record.get("timestamp")
    if timestamp is None:
        return False, "Invalid timestamp: null"

    return True, None


# ── Silver enrichment ──────────────────────────────────
def enrich_silver_record(record: dict) -> dict:
    """Add derived fields to a validated record.

    Args:
        record: Validated stock price record.

    Returns:
        Enriched record with price_change, price_change_pct, sector, volume_category.
    """
    enriched = dict(record)

    prev_close = record.get("prev_close")
    price = record["price"]

    if prev_close and prev_close > 0:
        enriched["price_change"] = round(price - prev_close, 4)
        enriched["price_change_pct"] = round((price - prev_close) / prev_close * 100, 4)
    else:
        enriched["price_change"] = None
        enriched["price_change_pct"] = None

    enriched["sector"] = SYMBOL_SECTORS.get(record.get("symbol"), "Unknown")
    enriched["volume_category"] = classify_volume(record.get("volume", 0))

    return enriched


# ── Quarantine ─────────────────────────────────────────
def quarantine_record(record: dict, reason: str) -> dict:
    """Wrap a rejected record with quarantine metadata.

    Args:
        record: The rejected record.
        reason: Why it was rejected.

    Returns:
        Record with quarantine_reason added.
    """
    quarantined = dict(record)
    quarantined["quarantine_reason"] = reason
    return quarantined


# ── Batch processing ───────────────────────────────────
def process_silver_batch(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split a batch into Silver (valid) and Quarantine (invalid).

    Args:
        records: List of stock price records.

    Returns:
        Tuple of (silver_records, quarantine_records).
    """
    silver = []
    quarantine = []

    for record in records:
        is_valid, reason = validate_record(record)
        if is_valid:
            silver.append(enrich_silver_record(record))
        else:
            quarantine.append(quarantine_record(record, reason))

    return silver, quarantine


# ── DLT table definitions (runs in Databricks only) ────
# Uncomment when running as a DLT pipeline:
#
# @dlt.table(
#     name=SILVER_TABLE_NAME,
#     comment="Cleaned and enriched stock prices with quality checks",
#     table_properties={"quality": "silver", "pipelines.autoOptimize.managed": "true"},
# )
# @dlt.expect_or_drop("valid_price", "price > 0")
# @dlt.expect_or_drop("valid_symbol", "symbol IS NOT NULL")
# @dlt.expect_or_drop("valid_volume", "volume >= 0")
# @dlt.expect_or_drop("valid_timestamp", "timestamp IS NOT NULL")
# def stock_prices_silver():
#     return (
#         dlt.read_stream(BRONZE_TABLE_NAME)
#         .withColumn("price_change", F.col("price") - F.col("prev_close"))
#         .withColumn("price_change_pct",
#             (F.col("price") - F.col("prev_close")) / F.col("prev_close") * 100
#         )
#         .withColumn("sector",
#             F.coalesce(
#                 F.map_from_entries(F.lit(list(SYMBOL_SECTORS.items())))[F.col("symbol")],
#                 F.lit("Unknown")
#             )
#         )
#         .withColumn("volume_category",
#             F.when(F.col("volume") >= 75_000_000, "very_high")
#              .when(F.col("volume") >= 25_000_000, "high")
#              .when(F.col("volume") >= 10_000_000, "medium")
#              .otherwise("low")
#         )
#     )
#
# @dlt.table(
#     name=QUARANTINE_TABLE_NAME,
#     comment="Rejected records that failed quality checks",
#     table_properties={"quality": "quarantine"},
# )
# def stock_prices_quarantine():
#     return (
#         dlt.read_stream(BRONZE_TABLE_NAME)
#         .filter(
#             (F.col("price").isNull()) | (F.col("price") <= 0) |
#             (F.col("symbol").isNull()) |
#             (F.col("volume").isNull()) | (F.col("volume") < 0) |
#             (F.col("timestamp").isNull())
#         )
#         .withColumn("quarantine_reason",
#             F.when(F.col("price").isNull() | (F.col("price") <= 0), "Invalid price")
#              .when(F.col("symbol").isNull(), "Invalid symbol")
#              .when(F.col("volume").isNull() | (F.col("volume") < 0), "Invalid volume")
#              .when(F.col("timestamp").isNull(), "Invalid timestamp")
#              .otherwise("Unknown")
#         )
#     )
