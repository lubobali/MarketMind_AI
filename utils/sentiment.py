"""MarketMind AI — Dual-model sentiment analysis engine.

Combines VADER (rule-based) and TextBlob (ML-based) for robust
financial sentiment scoring, with finance-specific keyword boosting.
"""

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Finance-specific keyword modifiers ───────────────────
# These adjust the raw score when domain-specific terms appear.
# Positive: earnings beats, upgrades, record revenue, etc.
# Negative: recalls, fraud, downgrades, SEC investigation, etc.
_POSITIVE_KEYWORDS = {
    "beats": 0.15,
    "surges": 0.15,
    "record revenue": 0.2,
    "record profits": 0.2,
    "all-time high": 0.2,
    "strong buy": 0.2,
    "upgrade": 0.15,
    "raises dividend": 0.15,
    "outperform": 0.1,
    "bullish": 0.15,
}

_NEGATIVE_KEYWORDS = {
    "recall": -0.15,
    "recalls": -0.15,
    "fraud": -0.25,
    "investigation": -0.15,
    "downgrade": -0.15,
    "plunges": -0.2,
    "crashes": -0.2,
    "massive loss": -0.2,
    "earnings miss": -0.15,
    "sell rating": -0.15,
    "bearish": -0.15,
    "bankruptcy": -0.25,
    "layoffs": -0.1,
    "defects": -0.15,
}

# Neutral dampeners — phrases that sound positive/negative to
# general-purpose models but are neutral in finance context.
_NEUTRAL_DAMPENERS = [
    "holds steady",
    "holds rates",
    "holds interest rates",
    "rates steady",
    "remains unchanged",
    "flat",
    "closed for holiday",
    "announces date",
]

# Singleton — VADER analyzer is stateless and thread-safe
_vader = SentimentIntensityAnalyzer()


def _finance_boost(text: str) -> float:
    """Calculate finance-specific keyword boost for a text.

    Scans for domain keywords and sums their modifiers.
    Clamped to [-0.5, 0.5] to avoid overwhelming the base models.
    """
    text_lower = text.lower()
    boost = 0.0

    for keyword, modifier in _POSITIVE_KEYWORDS.items():
        if keyword in text_lower:
            boost += modifier

    for keyword, modifier in _NEGATIVE_KEYWORDS.items():
        if keyword in text_lower:
            boost += modifier  # modifier is already negative

    return max(-0.5, min(0.5, boost))


def _score_to_label(score: float) -> str:
    """Convert a sentiment score to a human-readable label."""
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    return "neutral"


def analyze_sentiment(text: str | None) -> dict:
    """Analyze sentiment of a text using VADER + TextBlob + finance keywords.

    Args:
        text: The text to analyze. None or empty returns neutral.

    Returns:
        Dict with:
            sentiment_score: float in [-1.0, 1.0]
            sentiment_label: "positive" | "negative" | "neutral"
            confidence: float in [0.0, 1.0] — how much the two models agree
    """
    # Handle empty / None input
    if not text or not text.strip():
        return {
            "sentiment_score": 0.0,
            "sentiment_label": "neutral",
            "confidence": 1.0,
        }

    clean_text = text.strip()

    # ── VADER score (compound: -1 to 1) ──────────────────
    vader_scores = _vader.polarity_scores(clean_text)
    vader_compound = vader_scores["compound"]

    # ── TextBlob score (polarity: -1 to 1) ───────────────
    blob = TextBlob(clean_text)
    textblob_polarity = blob.sentiment.polarity

    # ── Finance keyword boost ────────────────────────────
    boost = _finance_boost(clean_text)

    # ── Neutral dampening ───────────────────────────────
    # Some phrases are neutral in finance but score positive/negative
    # in general-purpose models. Dampen toward zero.
    text_lower = clean_text.lower()
    dampen = any(phrase in text_lower for phrase in _NEUTRAL_DAMPENERS)

    # ── Combined score: weighted average + boost ─────────
    # VADER gets more weight — it handles social/news text better
    raw_score = (vader_compound * 0.6) + (textblob_polarity * 0.4) + boost
    if dampen:
        raw_score *= 0.3  # Push toward zero but preserve direction
    final_score = max(-1.0, min(1.0, raw_score))

    # ── Confidence: how much the two models agree ────────
    # If both models agree on direction, confidence is high.
    # If they disagree, confidence drops.
    agreement = 1.0 - abs(vader_compound - textblob_polarity)
    confidence = max(0.0, min(1.0, agreement))

    return {
        "sentiment_score": round(final_score, 4),
        "sentiment_label": _score_to_label(final_score),
        "confidence": round(confidence, 4),
    }
