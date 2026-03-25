# Databricks notebook source
# MAGIC %md
# MAGIC # MarketMind AI — Bronze Layer
# MAGIC Raw data landing zone. Reads JSON files from the producer,
# MAGIC adds ingestion metadata, and writes to Delta Live Table.
# MAGIC
# MAGIC **No transformations** — Bronze preserves raw data exactly as received.

# ── Imports (Spark + DLT available on Databricks) ──────
# These imports work in Databricks notebooks.
# For local testing, we use the pure-Python functions below.
# import dlt
# from pyspark.sql import functions as F

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── DLT table configuration ────────────────────────────
BRONZE_TABLE_NAME = "stock_prices_bronze"
BRONZE_TABLE_PROPERTIES = {
    "quality": "bronze",
    "pipelines.autoOptimize.managed": "true",
}

# ── Raw data path (set in Databricks or overridden) ────
RAW_PRICES_PATH = "/mnt/marketmind/raw/prices/"


# ── Pure-Python enrichment (testable without Spark) ────
def enrich_bronze_record(record: dict) -> dict:
    """Add ingestion metadata to a raw record.

    Args:
        record: Raw stock price record from the producer.

    Returns:
        Enriched record with ingest_time and trade_time added.
        All original fields preserved unchanged.
    """
    enriched = dict(record)

    # Add ingestion timestamp
    enriched["ingest_time"] = datetime.now(timezone.utc).isoformat()

    # Parse trade_time from raw timestamp
    raw_ts = record.get("timestamp")
    if raw_ts:
        enriched["trade_time"] = raw_ts
    else:
        enriched["trade_time"] = None

    return enriched


def enrich_bronze_batch(records: list[dict]) -> list[dict]:
    """Enrich a batch of raw records for Bronze landing.

    Args:
        records: List of raw stock price records.

    Returns:
        List of enriched records with metadata.
    """
    return [enrich_bronze_record(r) for r in records]


# ── DLT table definition (runs in Databricks only) ─────
# Uncomment when running as a DLT pipeline in Databricks:
#
# @dlt.table(
#     name=BRONZE_TABLE_NAME,
#     comment="Raw stock prices from Yahoo Finance — no transformations",
#     table_properties=BRONZE_TABLE_PROPERTIES,
# )
# def stock_prices_bronze():
#     return (
#         spark.readStream
#         .format("json")
#         .schema(STOCK_PRICE_RAW_SPARK_SCHEMA)
#         .option("maxFilesPerTrigger", 1)
#         .load(RAW_PRICES_PATH)
#         .withColumn("ingest_time", F.current_timestamp())
#         .withColumn("trade_time", F.to_timestamp(F.col("timestamp")))
#     )
