"""MarketMind AI — Configuration settings."""

# ── Tracked stock symbols by sector ─────────────────────
TRACKED_SYMBOLS = [
    # Technology
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    # Finance
    "JPM",
    "GS",
    "BAC",
    # Energy
    "XOM",
    "CVX",
    # Healthcare
    "PFE",
    "JNJ",
    # Consumer
    "TSLA",
    "WMT",
    "KO",
]

# ── Sector lookup ───────────────────────────────────────
SYMBOL_SECTORS = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "GOOGL": "Technology",
    "AMZN": "Technology",
    "JPM": "Finance",
    "GS": "Finance",
    "BAC": "Finance",
    "XOM": "Energy",
    "CVX": "Energy",
    "PFE": "Healthcare",
    "JNJ": "Healthcare",
    "TSLA": "Consumer",
    "WMT": "Consumer",
    "KO": "Consumer",
}

# ── Polling intervals (seconds) ────────────────────────
PRICE_POLL_INTERVAL = 30
NEWS_POLL_INTERVAL = 300  # 5 minutes

# ── Data paths (Databricks DBFS) ───────────────────────
RAW_PRICES_PATH = "/mnt/marketmind/raw/prices/"
RAW_NEWS_PATH = "/mnt/marketmind/raw/news/"
