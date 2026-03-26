# Databricks notebook source
# MAGIC %md
# MAGIC # MarketMind AI — News Producer
# MAGIC Fetches financial news from Yahoo Finance for tracked symbols
# MAGIC and writes them as JSON files for Spark Structured Streaming.
# MAGIC
# MAGIC **Data flow**: yfinance `.news` → normalize → newline-delimited JSON → DBFS

import json
import logging
import os
from datetime import datetime, timezone

import yfinance as yf

logger = logging.getLogger(__name__)


def normalize_yfinance_article(raw_article: dict, symbol: str) -> dict | None:
    """Flatten a nested yfinance news article into our clean schema.

    Args:
        raw_article: Raw article dict from yfinance Ticker.news.
        symbol: The ticker symbol this article was fetched for.

    Returns:
        Normalized dict matching NEWS_RAW_SCHEMA, or None if malformed.
    """
    content = raw_article.get("content")
    if content is None:
        logger.warning("Article missing 'content' key — skipping")
        return None

    title = content.get("title")
    if title is None:
        logger.warning("Article missing 'title' — skipping")
        return None

    # Extract provider name — may be None or missing
    provider = content.get("provider")
    if provider and isinstance(provider, dict):
        source = provider.get("displayName", "Unknown")
    else:
        source = "Unknown"

    # Extract canonical URL — may be None or missing
    canonical = content.get("canonicalUrl")
    if canonical and isinstance(canonical, dict):
        url = canonical.get("url", "")
    else:
        url = ""

    return {
        "headline": title,
        "summary": content.get("summary", ""),
        "source": source,
        "url": url,
        "symbols": [symbol],
        "published_at": content.get("pubDate", ""),
    }


def fetch_news_for_symbols(symbols: list[str]) -> list[dict]:
    """Fetch and normalize news articles for a list of stock symbols.

    Deduplicates articles by headline — the same article often appears
    across multiple tickers (e.g., market-wide news).

    Args:
        symbols: List of ticker symbols (e.g., ["AAPL", "NVDA"]).

    Returns:
        List of normalized news records. Failed symbols are skipped.
    """
    if not symbols:
        return []

    articles = []
    seen_headlines = set()

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            raw_news = ticker.news

            if not raw_news:
                logger.info("No news for %s", symbol)
                continue

            for raw_article in raw_news:
                normalized = normalize_yfinance_article(raw_article, symbol)
                if normalized is None:
                    continue

                # Deduplicate by headline
                headline = normalized["headline"]
                if headline in seen_headlines:
                    continue
                seen_headlines.add(headline)

                articles.append(normalized)

        except Exception as e:
            logger.warning("Failed to fetch news for %s: %s", symbol, e)

    return articles


def write_news_to_json(records: list[dict], output_dir: str) -> str:
    """Write news records as a newline-delimited JSON file.

    Args:
        records: List of normalized news records.
        output_dir: Directory to write the JSON file into.

    Returns:
        Path to the written file.
    """
    os.makedirs(output_dir, exist_ok=True)

    filename = f"news_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Wrote %d news records to %s", len(records), filepath)
    return filepath


# ── Main loop (runs in Databricks notebook) ─────────────
if __name__ == "__main__":
    import time

    from config.settings import NEWS_POLL_INTERVAL, RAW_NEWS_PATH, TRACKED_SYMBOLS

    print(f"MarketMind AI — Fetching news for {len(TRACKED_SYMBOLS)} symbols every {NEWS_POLL_INTERVAL}s")

    while True:
        news = fetch_news_for_symbols(TRACKED_SYMBOLS)
        if news:
            path = write_news_to_json(news, RAW_NEWS_PATH)
            print(f"[{datetime.now(timezone.utc).isoformat()}] Wrote {len(news)} articles → {path}")
        else:
            print(f"[{datetime.now(timezone.utc).isoformat()}] No news fetched")

        time.sleep(NEWS_POLL_INTERVAL)
