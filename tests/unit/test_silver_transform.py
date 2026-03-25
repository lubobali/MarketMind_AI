"""Tests for Silver layer — clean, validate, and enrich.

Requirements:
- Data quality: reject records with null/invalid price, volume, symbol, timestamp
- Enrich with: price_change, price_change_pct, sector, volume_category
- Quarantine rejected records with a reason
- DLT expectations map to our quality rules
"""

import pytest


# ── Test: Silver enrichment calculations ────────────────
class TestSilverEnrichment:
    """Silver must calculate derived fields correctly."""

    def test_price_change(self, sample_stock_record):
        """price_change = price - prev_close."""
        from notebooks.silver_transform import enrich_silver_record

        result = enrich_silver_record(sample_stock_record)

        expected = sample_stock_record["price"] - sample_stock_record["prev_close"]
        assert result["price_change"] == pytest.approx(expected, abs=0.01)

    def test_price_change_pct(self, sample_stock_record):
        """price_change_pct = (price - prev_close) / prev_close * 100."""
        from notebooks.silver_transform import enrich_silver_record

        result = enrich_silver_record(sample_stock_record)

        expected = (
            (sample_stock_record["price"] - sample_stock_record["prev_close"]) / sample_stock_record["prev_close"] * 100
        )
        assert result["price_change_pct"] == pytest.approx(expected, abs=0.01)

    def test_sector_lookup(self, sample_stock_record):
        """Symbol must be mapped to its sector."""
        from notebooks.silver_transform import enrich_silver_record

        result = enrich_silver_record(sample_stock_record)

        assert result["sector"] == "Technology"  # NVDA

    @pytest.mark.parametrize(
        "volume,expected_category",
        [
            (100_000_000, "very_high"),
            (50_000_000, "high"),
            (15_000_000, "medium"),
            (3_000_000, "low"),
        ],
    )
    def test_volume_category(self, volume, expected_category):
        """Volume must be categorized into buckets."""
        from notebooks.silver_transform import classify_volume

        result = classify_volume(volume)
        assert result == expected_category

    def test_preserves_raw_fields(self, sample_stock_record):
        """Silver must keep all raw fields from Bronze."""
        from notebooks.silver_transform import enrich_silver_record

        result = enrich_silver_record(sample_stock_record)

        for key in sample_stock_record:
            assert key in result, f"Raw field '{key}' dropped at Silver"


# ── Test: Data quality validation ───────────────────────
class TestSilverDataQuality:
    """Silver must reject bad records."""

    def test_valid_record_passes(self, sample_stock_record):
        """A good record should pass all quality checks."""
        from notebooks.silver_transform import validate_record

        is_valid, reason = validate_record(sample_stock_record)
        assert is_valid is True
        assert reason is None

    def test_null_price_fails(self, sample_stock_record):
        """Null price must be rejected."""
        from notebooks.silver_transform import validate_record

        bad = dict(sample_stock_record)
        bad["price"] = None

        is_valid, reason = validate_record(bad)
        assert is_valid is False
        assert "price" in reason.lower()

    def test_negative_price_fails(self, sample_stock_record):
        """Negative price must be rejected."""
        from notebooks.silver_transform import validate_record

        bad = dict(sample_stock_record)
        bad["price"] = -5.0

        is_valid, reason = validate_record(bad)
        assert is_valid is False
        assert "price" in reason.lower()

    def test_zero_price_fails(self, sample_stock_record):
        """Zero price must be rejected."""
        from notebooks.silver_transform import validate_record

        bad = dict(sample_stock_record)
        bad["price"] = 0

        is_valid, reason = validate_record(bad)
        assert is_valid is False

    def test_null_symbol_fails(self, sample_stock_record):
        """Null symbol must be rejected."""
        from notebooks.silver_transform import validate_record

        bad = dict(sample_stock_record)
        bad["symbol"] = None

        is_valid, reason = validate_record(bad)
        assert is_valid is False
        assert "symbol" in reason.lower()

    def test_null_volume_fails(self, sample_stock_record):
        """Null volume must be rejected."""
        from notebooks.silver_transform import validate_record

        bad = dict(sample_stock_record)
        bad["volume"] = None

        is_valid, reason = validate_record(bad)
        assert is_valid is False
        assert "volume" in reason.lower()

    def test_negative_volume_fails(self, sample_stock_record):
        """Negative volume must be rejected."""
        from notebooks.silver_transform import validate_record

        bad = dict(sample_stock_record)
        bad["volume"] = -100

        is_valid, reason = validate_record(bad)
        assert is_valid is False

    def test_null_timestamp_fails(self, sample_stock_record):
        """Null timestamp must be rejected."""
        from notebooks.silver_transform import validate_record

        bad = dict(sample_stock_record)
        bad["timestamp"] = None

        is_valid, reason = validate_record(bad)
        assert is_valid is False
        assert "timestamp" in reason.lower()


# ── Test: Quarantine ────────────────────────────────────
class TestSilverQuarantine:
    """Bad records must go to quarantine with a reason."""

    def test_quarantine_record_has_reason(self, sample_stock_record):
        """Quarantined record must include the rejection reason."""
        from notebooks.silver_transform import quarantine_record

        bad = dict(sample_stock_record)
        bad["price"] = None

        result = quarantine_record(bad, "Invalid price: null")

        assert result["quarantine_reason"] == "Invalid price: null"
        assert result["symbol"] == sample_stock_record["symbol"]

    def test_quarantine_preserves_original_data(self, sample_stock_record):
        """Quarantine must keep the original record for debugging."""
        from notebooks.silver_transform import quarantine_record

        bad = dict(sample_stock_record)
        bad["price"] = -5.0

        result = quarantine_record(bad, "Invalid price: negative")

        assert result["price"] == -5.0
        assert result["symbol"] == sample_stock_record["symbol"]


# ── Test: Process batch (Silver + Quarantine split) ─────
class TestSilverBatchProcessing:
    """Batch must split into valid (Silver) and invalid (Quarantine)."""

    def test_splits_valid_and_invalid(self):
        """Good records → Silver, bad records → Quarantine."""
        from notebooks.silver_transform import process_silver_batch

        records = [
            {
                "symbol": "AAPL",
                "price": 178.52,
                "open": 178.0,
                "high": 179.1,
                "low": 177.8,
                "prev_close": 177.5,
                "volume": 52_100_000,
                "market_cap": 2_800_000_000_000,
                "timestamp": "2026-03-24T14:30:00Z",
            },
            {
                "symbol": None,
                "price": 100.0,
                "open": 99.0,
                "high": 101.0,
                "low": 98.0,
                "prev_close": 99.5,
                "volume": 1_000_000,
                "market_cap": 500_000_000_000,
                "timestamp": "2026-03-24T14:30:00Z",
            },
            {
                "symbol": "NVDA",
                "price": None,
                "open": 870.0,
                "high": 880.0,
                "low": 868.0,
                "prev_close": 872.3,
                "volume": 45_000_000,
                "market_cap": 2_150_000_000_000,
                "timestamp": "2026-03-24T14:30:00Z",
            },
        ]

        silver, quarantine = process_silver_batch(records)

        assert len(silver) == 1
        assert silver[0]["symbol"] == "AAPL"
        assert len(quarantine) == 2

    def test_all_valid_returns_empty_quarantine(self, sample_stock_batch):
        """If all records are valid, quarantine should be empty."""
        from notebooks.silver_transform import process_silver_batch

        silver, quarantine = process_silver_batch(sample_stock_batch)

        assert len(silver) == len(sample_stock_batch)
        assert len(quarantine) == 0

    def test_silver_records_are_enriched(self, sample_stock_batch):
        """Silver output must have derived fields."""
        from notebooks.silver_transform import process_silver_batch

        silver, _ = process_silver_batch(sample_stock_batch)

        for record in silver:
            assert "price_change" in record
            assert "price_change_pct" in record
            assert "sector" in record
            assert "volume_category" in record


# ── Test: DLT configuration ────────────────────────────
class TestSilverDLTConfig:
    def test_table_name(self):
        from notebooks.silver_transform import SILVER_TABLE_NAME

        assert SILVER_TABLE_NAME == "stock_prices_silver"

    def test_quarantine_table_name(self):
        from notebooks.silver_transform import QUARANTINE_TABLE_NAME

        assert QUARANTINE_TABLE_NAME == "stock_prices_quarantine"

    def test_quality_expectations(self):
        """DLT expectations must match our validation rules."""
        from notebooks.silver_transform import DLT_EXPECTATIONS

        assert "valid_price" in DLT_EXPECTATIONS
        assert "valid_symbol" in DLT_EXPECTATIONS
        assert "valid_volume" in DLT_EXPECTATIONS
        assert "valid_timestamp" in DLT_EXPECTATIONS
