# Databricks notebook source
# MAGIC %md
# MAGIC # MarketMind AI — News NLP Pipeline
# MAGIC Cleans text, scores sentiment per article, aggregates per symbol,
# MAGIC and computes overall market mood.
# MAGIC
# MAGIC **Data flow**: Raw news → clean → sentiment → symbol agg → market mood

import html
import logging
import re
from collections import defaultdict

from utils.sentiment import analyze_sentiment

logger = logging.getLogger(__name__)

# ── HTML tag pattern (compiled once) ─────────────────────
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    """Strip HTML tags, decode entities, normalize whitespace.

    Args:
        text: Raw text that may contain HTML markup.

    Returns:
        Clean plain text. Empty string if input is None or empty.
    """
    if not text:
        return ""

    # Decode HTML entities first (e.g., &amp; → &)
    cleaned = html.unescape(text)

    # Strip HTML tags
    cleaned = _HTML_TAG_RE.sub("", cleaned)

    # Normalize whitespace (tabs, newlines, multiple spaces → single space)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)

    return cleaned.strip()


def score_article(article: dict) -> dict:
    """Score a news article's sentiment using headline + summary.

    Combines headline and summary text for a fuller signal.
    Cleans HTML from both before scoring. Does not mutate the input.

    Args:
        article: Normalized news record with headline and summary.

    Returns:
        New dict with all original fields plus sentiment_score,
        sentiment_label, and confidence.
    """
    headline = clean_text(article.get("headline", ""))
    summary = clean_text(article.get("summary", ""))

    # Combine for richer signal — headline carries more weight
    # but summary adds context
    combined = f"{headline}. {summary}".strip(". ")

    sentiment = analyze_sentiment(combined)

    scored = dict(article)
    scored["sentiment_score"] = sentiment["sentiment_score"]
    scored["sentiment_label"] = sentiment["sentiment_label"]
    scored["confidence"] = sentiment["confidence"]

    return scored


def compute_symbol_sentiment(scored_articles: list[dict]) -> dict:
    """Aggregate sentiment across articles for each symbol.

    Args:
        scored_articles: List of articles with sentiment_score and symbols.

    Returns:
        Dict keyed by symbol, each with:
            avg_sentiment, article_count, most_positive, most_negative
    """
    if not scored_articles:
        return {}

    # Group articles by symbol
    by_symbol = defaultdict(list)
    for article in scored_articles:
        for symbol in article.get("symbols", []):
            by_symbol[symbol].append(article)

    result = {}
    for symbol, articles in by_symbol.items():
        scores = [a["sentiment_score"] for a in articles]
        avg = sum(scores) / len(scores)

        sorted_by_score = sorted(articles, key=lambda a: a["sentiment_score"])
        most_negative = sorted_by_score[0]
        most_positive = sorted_by_score[-1]

        result[symbol] = {
            "avg_sentiment": round(avg, 4),
            "article_count": len(articles),
            "most_positive": most_positive.get("headline", ""),
            "most_negative": most_negative.get("headline", ""),
        }

    return result


def compute_market_mood(scored_articles: list[dict]) -> dict:
    """Compute overall market mood from all scored articles.

    Args:
        scored_articles: List of articles with sentiment_score and sentiment_label.

    Returns:
        Dict with mood ("bullish"/"bearish"/"neutral"), avg_score,
        article_count, positive_count, negative_count, neutral_count.
    """
    if not scored_articles:
        return {
            "mood": "neutral",
            "avg_score": 0.0,
            "article_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
        }

    scores = [a["sentiment_score"] for a in scored_articles]
    avg = sum(scores) / len(scores)

    positive_count = sum(1 for a in scored_articles if a["sentiment_label"] == "positive")
    negative_count = sum(1 for a in scored_articles if a["sentiment_label"] == "negative")
    neutral_count = sum(1 for a in scored_articles if a["sentiment_label"] == "neutral")

    if avg > 0.2:
        mood = "bullish"
    elif avg < -0.2:
        mood = "bearish"
    else:
        mood = "neutral"

    return {
        "mood": mood,
        "avg_score": round(avg, 4),
        "article_count": len(scored_articles),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
    }
