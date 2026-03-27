"""MarketMind AI — Schema definitions.

These mirror the Spark StructType schemas used in Databricks notebooks.
Defined here as plain Python so unit/integration tests can validate
data shape without needing a Spark session.
"""

from dataclasses import dataclass


@dataclass
class SchemaField:
    """Represents a single field in a schema."""

    name: str
    type: str
    nullable: bool = True


class Schema:
    """Lightweight schema definition for testing without Spark."""

    def __init__(self, fields: list[SchemaField]):
        self.fields = fields

    @property
    def field_names(self) -> set[str]:
        return {f.name for f in self.fields}


# ── Stock price raw schema ──────────────────────────────
STOCK_PRICE_RAW_SCHEMA = Schema(
    [
        SchemaField("symbol", "string", nullable=False),
        SchemaField("price", "double"),
        SchemaField("open", "double"),
        SchemaField("high", "double"),
        SchemaField("low", "double"),
        SchemaField("prev_close", "double"),
        SchemaField("volume", "long"),
        SchemaField("market_cap", "long"),
        SchemaField("timestamp", "string"),
    ]
)

# ── News raw schema ────────────────────────────────────
NEWS_RAW_SCHEMA = Schema(
    [
        SchemaField("headline", "string"),
        SchemaField("summary", "string"),
        SchemaField("source", "string"),
        SchemaField("url", "string"),
        SchemaField("symbols", "array<string>"),
        SchemaField("published_at", "string"),
    ]
)

# ── News scored schema (Silver) ───────────────────────
NEWS_SCORED_SCHEMA = Schema(
    [
        SchemaField("headline", "string"),
        SchemaField("summary", "string"),
        SchemaField("source", "string"),
        SchemaField("url", "string"),
        SchemaField("symbols", "array<string>"),
        SchemaField("published_at", "string"),
        SchemaField("sentiment_score", "double"),
        SchemaField("sentiment_label", "string"),
        SchemaField("confidence", "double"),
    ]
)

# ── Symbol sentiment aggregation schema (Gold) ────────
SYMBOL_SENTIMENT_AGG_SCHEMA = Schema(
    [
        SchemaField("symbol", "string", nullable=False),
        SchemaField("avg_sentiment", "double"),
        SchemaField("article_count", "long"),
        SchemaField("most_positive", "string"),
        SchemaField("most_negative", "string"),
    ]
)

# ── Market mood schema (Gold) ─────────────────────────
MARKET_MOOD_SCHEMA = Schema(
    [
        SchemaField("mood", "string"),
        SchemaField("avg_score", "double"),
        SchemaField("article_count", "long"),
        SchemaField("positive_count", "long"),
        SchemaField("negative_count", "long"),
        SchemaField("neutral_count", "long"),
    ]
)


# ══════════════════════════════════════════════════════════════
#  Phase 5 — Advanced Analytics schemas
# ══════════════════════════════════════════════════════════════

# ── Technical indicators per symbol ─────────────────────────
TECHNICAL_INDICATORS_SCHEMA = Schema(
    [
        SchemaField("symbol", "string", nullable=False),
        SchemaField("sector", "string"),
        SchemaField("latest_date", "date"),
        SchemaField("latest_close", "double"),
        SchemaField("rsi", "double"),
        SchemaField("macd_line", "double"),
        SchemaField("macd_signal", "double"),
        SchemaField("macd_histogram", "double"),
        SchemaField("bb_upper", "double"),
        SchemaField("bb_middle", "double"),
        SchemaField("bb_lower", "double"),
    ]
)

# ── Market signals (buy/sell/hold) ──────────────────────────
MARKET_SIGNALS_SCHEMA = Schema(
    [
        SchemaField("symbol", "string", nullable=False),
        SchemaField("sector", "string"),
        SchemaField("latest_date", "date"),
        SchemaField("latest_close", "double"),
        SchemaField("rsi", "double"),
        SchemaField("macd_line", "double"),
        SchemaField("macd_signal", "double"),
        SchemaField("macd_histogram", "double"),
        SchemaField("bb_upper", "double"),
        SchemaField("bb_middle", "double"),
        SchemaField("bb_lower", "double"),
        SchemaField("signal", "string"),
    ]
)

# ── Moving averages + VWAP ──────────────────────────────────
MOVING_AVERAGES_SCHEMA = Schema(
    [
        SchemaField("symbol", "string", nullable=False),
        SchemaField("sector", "string"),
        SchemaField("date", "date"),
        SchemaField("day_close", "double"),
        SchemaField("total_volume", "long"),
        SchemaField("sma_5", "double"),
        SchemaField("sma_20", "double"),
        SchemaField("sma_50", "double"),
        SchemaField("vwap_20", "double"),
    ]
)

# ── Sector rankings ─────────────────────────────────────────
SECTOR_RANKINGS_SCHEMA = Schema(
    [
        SchemaField("symbol", "string", nullable=False),
        SchemaField("sector", "string"),
        SchemaField("date", "date"),
        SchemaField("avg_change_pct", "double"),
        SchemaField("sector_rank", "integer"),
        SchemaField("sector_size", "integer"),
        SchemaField("percentile", "double"),
    ]
)

# ── Volume spikes ───────────────────────────────────────────
VOLUME_SPIKES_SCHEMA = Schema(
    [
        SchemaField("symbol", "string", nullable=False),
        SchemaField("sector", "string"),
        SchemaField("date", "date"),
        SchemaField("total_volume", "long"),
        SchemaField("avg_volume_20d", "long"),
        SchemaField("volume_ratio", "double"),
        SchemaField("is_spike", "boolean"),
    ]
)
