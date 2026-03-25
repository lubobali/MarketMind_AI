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
