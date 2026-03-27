# Databricks notebook source
# MAGIC %md
# MAGIC # MarketMind AI — News NLP Pipeline (DLT)
# MAGIC Bronze → Silver → Gold for financial news with sentiment analysis.
# MAGIC
# MAGIC **Run this as a Delta Live Tables pipeline, NOT as a regular notebook.**
# MAGIC
# MAGIC To set up:
# MAGIC 1. Go to Pipelines → Create Pipeline (or add to existing)
# MAGIC 2. Add this notebook as an additional source
# MAGIC 3. Same target schema: "marketmind"
# MAGIC 4. Click Start
# MAGIC
# MAGIC **Tables created:**
# MAGIC - `stock_news_bronze` — Raw news articles from Yahoo Finance
# MAGIC - `news_sentiment_silver` — Cleaned + sentiment scored
# MAGIC - `symbol_sentiment_agg` — Per-symbol sentiment aggregation
# MAGIC - `market_mood` — Overall market sentiment snapshot

# COMMAND ----------

import dlt
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
)

# COMMAND ----------

# ── Configuration ──────────────────────────────────────

RAW_NEWS_TABLE = "bootcamp_students.lubo_marketmind_ai.raw_stock_news"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze Layer — Raw News Landing

# COMMAND ----------

@dlt.table(
    name="stock_news_bronze",
    comment="Raw financial news articles from Yahoo Finance — no transformations",
    table_properties={
        "quality": "bronze",
        "pipelines.autoOptimize.managed": "true",
    },
)
def stock_news_bronze():
    return (
        spark.readStream
        .table(RAW_NEWS_TABLE)
        .withColumn("ingest_time", F.current_timestamp())
        .withColumn("pub_time", F.to_timestamp(F.col("published_at")))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver Layer — Clean + Sentiment Score
# MAGIC
# MAGIC Uses a UDF to run VADER + TextBlob dual-model sentiment analysis.
# MAGIC HTML is stripped, whitespace normalized, then scored.

# COMMAND ----------

# ── Sentiment Pandas UDF ─────────────────────────────────────
# Pandas UDF instantiates VADER once per batch (not per row) for efficiency.
# Returns a struct with sentiment_score, sentiment_label, confidence.

SENTIMENT_SCHEMA = StructType([
    StructField("sentiment_score", DoubleType()),
    StructField("sentiment_label", StringType()),
    StructField("confidence", DoubleType()),
])

@F.pandas_udf(SENTIMENT_SCHEMA, F.PandasUDFType.SCALAR)
def score_text_udf(headline_series: pd.Series, summary_series: pd.Series) -> pd.DataFrame:
    """Score headline + summary with dual-model sentiment (VADER + TextBlob).

    Uses pandas_udf so VADER analyzer is created once per batch, not per row.
    """
    import html as html_mod
    import re

    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    # Instantiate ONCE per batch — this is the key performance improvement
    vader = SentimentIntensityAnalyzer()

    # ── Finance keyword modifiers ────────────────────────
    POSITIVE_KW = {
        "beats": 0.15, "surges": 0.15, "record revenue": 0.2,
        "record profits": 0.2, "all-time high": 0.2, "strong buy": 0.2,
        "upgrade": 0.15, "raises dividend": 0.15, "outperform": 0.1,
        "bullish": 0.15,
    }
    NEGATIVE_KW = {
        "recall": -0.15, "recalls": -0.15, "fraud": -0.25,
        "investigation": -0.15, "downgrade": -0.15, "plunges": -0.2,
        "crashes": -0.2, "massive loss": -0.2, "earnings miss": -0.15,
        "sell rating": -0.15, "bearish": -0.15, "bankruptcy": -0.25,
        "layoffs": -0.1, "defects": -0.15,
    }
    NEUTRAL_DAMPENERS = [
        "holds steady", "holds rates", "holds interest rates",
        "rates steady", "remains unchanged", "flat",
        "closed for holiday", "announces date",
    ]

    def _clean(text):
        if not text or (isinstance(text, float)):
            return ""
        text = html_mod.unescape(str(text))
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _score_one(headline, summary):
        h = _clean(headline)
        s = _clean(summary)
        combined = f"{h}. {s}".strip(". ")

        if not combined:
            return (0.0, "neutral", 1.0)

        vader_compound = vader.polarity_scores(combined)["compound"]
        blob_polarity = TextBlob(combined).sentiment.polarity

        text_lower = combined.lower()
        boost = 0.0
        for kw, mod in POSITIVE_KW.items():
            if kw in text_lower:
                boost += mod
        for kw, mod in NEGATIVE_KW.items():
            if kw in text_lower:
                boost += mod
        boost = max(-0.5, min(0.5, boost))

        dampen = any(p in text_lower for p in NEUTRAL_DAMPENERS)

        raw = (vader_compound * 0.6) + (blob_polarity * 0.4) + boost
        if dampen:
            raw *= 0.3
        final = max(-1.0, min(1.0, raw))

        confidence = max(0.0, min(1.0, 1.0 - abs(vader_compound - blob_polarity)))

        if final > 0.05:
            label = "positive"
        elif final < -0.05:
            label = "negative"
        else:
            label = "neutral"

        return (round(final, 4), label, round(confidence, 4))

    results = [_score_one(h, s) for h, s in zip(headline_series, summary_series)]
    return pd.DataFrame(results, columns=["sentiment_score", "sentiment_label", "confidence"])

# COMMAND ----------

@dlt.table(
    name="news_sentiment_silver",
    comment="News articles with sentiment scores — cleaned and validated",
    table_properties={
        "quality": "silver",
        "pipelines.autoOptimize.managed": "true",
    },
)
@dlt.expect_or_drop("valid_headline", "headline IS NOT NULL")
@dlt.expect_or_drop("valid_published", "published_at IS NOT NULL")
def news_sentiment_silver():
    scored = (
        dlt.read_stream("stock_news_bronze")
        .withColumn("_sentiment", score_text_udf(F.col("headline"), F.col("summary")))
    )
    return (
        scored
        .withColumn("sentiment_score", F.col("_sentiment.sentiment_score"))
        .withColumn("sentiment_label", F.col("_sentiment.sentiment_label"))
        .withColumn("confidence", F.col("_sentiment.confidence"))
        .drop("_sentiment")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold Layer — Aggregated Sentiment

# COMMAND ----------

@dlt.table(
    name="symbol_sentiment_agg",
    comment="Per-symbol sentiment aggregation — avg score, article count, extremes",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true",
    },
)
def symbol_sentiment_agg():
    return (
        dlt.read("news_sentiment_silver")
        .withColumn("symbol", F.explode("symbols"))
        .groupBy("symbol")
        .agg(
            F.avg("sentiment_score").alias("avg_sentiment"),
            F.count("*").alias("article_count"),
            F.max("sentiment_score").alias("max_sentiment"),
            F.min("sentiment_score").alias("min_sentiment"),
        )
    )

# COMMAND ----------

@dlt.table(
    name="market_mood",
    comment="Overall market sentiment snapshot",
    table_properties={
        "quality": "gold",
    },
)
def market_mood():
    return (
        dlt.read("news_sentiment_silver")
        .agg(
            F.avg("sentiment_score").alias("avg_score"),
            F.count("*").alias("article_count"),
            F.sum(F.when(F.col("sentiment_label") == "positive", 1).otherwise(0)).alias("positive_count"),
            F.sum(F.when(F.col("sentiment_label") == "negative", 1).otherwise(0)).alias("negative_count"),
            F.sum(F.when(F.col("sentiment_label") == "neutral", 1).otherwise(0)).alias("neutral_count"),
        )
        .withColumn("mood",
            F.when(F.col("avg_score") > 0.2, "bullish")
             .when(F.col("avg_score") < -0.2, "bearish")
             .otherwise("neutral")
        )
    )
