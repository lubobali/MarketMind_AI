# Databricks notebook source
# MAGIC %md
# MAGIC # MarketMind AI — News Producer
# MAGIC Fetches financial news from Yahoo Finance for all 15 tracked symbols
# MAGIC and writes to a Delta table for the DLT news pipeline to consume.
# MAGIC
# MAGIC **Run this BEFORE the news DLT pipeline.**

# COMMAND ----------

# MAGIC %pip install yfinance
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

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

RAW_NEWS_TABLE = "bootcamp_students.lubo_marketmind_ai.raw_stock_news"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch News from Yahoo Finance

# COMMAND ----------

def fetch_all_news(symbols):
    """Fetch and normalize news for all tracked symbols.

    Deduplicates by headline — same article often appears
    across multiple tickers.
    """
    articles = []
    seen_headlines = set()

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            raw_news = ticker.news or []

            for item in raw_news:
                content = item.get("content", {})
                title = content.get("title")

                if not title or title in seen_headlines:
                    continue
                seen_headlines.add(title)

                provider = content.get("provider") or {}
                canonical = content.get("canonicalUrl") or {}

                articles.append({
                    "headline": title,
                    "summary": content.get("summary", ""),
                    "source": provider.get("displayName", "Unknown"),
                    "url": canonical.get("url", ""),
                    "symbols": [symbol],
                    "published_at": content.get("pubDate", ""),
                })
                print(f"  ✅ [{symbol}] {title[:60]}...")

        except Exception as e:
            print(f"  ❌ {symbol}: {e}")

    return articles

# COMMAND ----------

print(f"🚀 Fetching news for {len(TRACKED_SYMBOLS)} symbols...")
articles = fetch_all_news(TRACKED_SYMBOLS)
print(f"\n📰 Got {len(articles)} unique articles")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Delta Table

# COMMAND ----------

from pyspark.sql.types import (
    ArrayType,
    StringType,
    StructField,
    StructType,
)

NEWS_SCHEMA = StructType([
    StructField("headline", StringType(), nullable=True),
    StructField("summary", StringType(), nullable=True),
    StructField("source", StringType(), nullable=True),
    StructField("url", StringType(), nullable=True),
    StructField("symbols", ArrayType(StringType()), nullable=True),
    StructField("published_at", StringType(), nullable=True),
])

# COMMAND ----------

if articles:
    df = spark.createDataFrame(articles, schema=NEWS_SCHEMA)
    df.write.format("delta").mode("overwrite").saveAsTable(RAW_NEWS_TABLE)
    print(f"✅ Wrote {len(articles)} articles → {RAW_NEWS_TABLE}")
else:
    print("❌ No articles fetched — check yfinance connectivity")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify the Data

# COMMAND ----------

df = spark.table(RAW_NEWS_TABLE)
print(f"Schema: {df.schema.simpleString()}")
print(f"Record count: {df.count()}")
display(df)
