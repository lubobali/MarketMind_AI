"""Tests for Bronze layer — raw data landing.

Requirements:
- Read raw JSON price records (as produced by kafka_producer)
- Add ingest_timestamp to every record
- Parse string timestamp to proper datetime
- Keep ALL raw fields — no transformation at Bronze
- Define DLT table with correct name and properties
"""

from datetime import datetime, timezone


# ── Test: Bronze record enrichment ──────────────────────
class TestBronzeEnrichment:
    """Bronze layer must add metadata without changing raw data."""

    def test_adds_ingest_timestamp(self, sample_stock_record):
        """Every Bronze record must have an ingest_timestamp."""
        from notebooks.bronze_streaming import enrich_bronze_record

        result = enrich_bronze_record(sample_stock_record)

        assert "ingest_time" in result
        assert result["ingest_time"] is not None

    def test_ingest_timestamp_is_recent(self, sample_stock_record):
        """Ingest timestamp must be current time, not the data timestamp."""
        from notebooks.bronze_streaming import enrich_bronze_record

        before = datetime.now(timezone.utc)
        result = enrich_bronze_record(sample_stock_record)
        after = datetime.now(timezone.utc)

        ingest = datetime.fromisoformat(result["ingest_time"].replace("Z", "+00:00"))
        assert before <= ingest <= after

    def test_preserves_all_raw_fields(self, sample_stock_record):
        """Bronze must not drop or rename any raw fields."""
        from notebooks.bronze_streaming import enrich_bronze_record

        result = enrich_bronze_record(sample_stock_record)

        for key in sample_stock_record:
            assert key in result, f"Raw field '{key}' was dropped at Bronze"
            assert result[key] == sample_stock_record[key], f"Raw field '{key}' was modified at Bronze"

    def test_parses_trade_time_from_timestamp(self, sample_stock_record):
        """Bronze must parse the string timestamp into trade_time."""
        from notebooks.bronze_streaming import enrich_bronze_record

        result = enrich_bronze_record(sample_stock_record)

        assert "trade_time" in result
        # Must be parseable as datetime
        parsed = datetime.fromisoformat(result["trade_time"].replace("Z", "+00:00"))
        assert parsed.year == 2026


# ── Test: Bronze handles batch of records ───────────────
class TestBronzeBatch:
    """Bronze must process batches correctly."""

    def test_enriches_all_records_in_batch(self, sample_stock_batch):
        """Every record in a batch gets enriched."""
        from notebooks.bronze_streaming import enrich_bronze_batch

        results = enrich_bronze_batch(sample_stock_batch)

        assert len(results) == len(sample_stock_batch)
        for record in results:
            assert "ingest_time" in record
            assert "trade_time" in record

    def test_batch_preserves_symbol_order(self, sample_stock_batch):
        """Output order matches input order."""
        from notebooks.bronze_streaming import enrich_bronze_batch

        results = enrich_bronze_batch(sample_stock_batch)

        input_symbols = [r["symbol"] for r in sample_stock_batch]
        output_symbols = [r["symbol"] for r in results]
        assert input_symbols == output_symbols


# ── Test: DLT table configuration ──────────────────────
class TestBronzeDLTConfig:
    """Bronze DLT table must have correct metadata."""

    def test_table_name(self):
        from notebooks.bronze_streaming import BRONZE_TABLE_NAME

        assert BRONZE_TABLE_NAME == "stock_prices_bronze"

    def test_table_properties(self):
        from notebooks.bronze_streaming import BRONZE_TABLE_PROPERTIES

        assert BRONZE_TABLE_PROPERTIES["quality"] == "bronze"
        assert "pipelines.autoOptimize.managed" in BRONZE_TABLE_PROPERTIES


# ── Test: handles malformed records ─────────────────────
class TestBronzeEdgeCases:
    """Bronze must not crash on bad data."""

    def test_missing_timestamp_uses_ingest_time(self):
        """If raw record has no timestamp, trade_time should be None."""
        from notebooks.bronze_streaming import enrich_bronze_record

        record = {
            "symbol": "AAPL",
            "price": 178.52,
            "open": 178.00,
            "high": 179.10,
            "low": 177.80,
            "prev_close": 177.50,
            "volume": 52_100_000,
            "market_cap": 2_800_000_000_000,
            "timestamp": None,
        }

        result = enrich_bronze_record(record)

        assert result["trade_time"] is None
        assert result["ingest_time"] is not None

    def test_empty_batch_returns_empty(self):
        """Empty input should return empty output."""
        from notebooks.bronze_streaming import enrich_bronze_batch

        results = enrich_bronze_batch([])
        assert results == []
