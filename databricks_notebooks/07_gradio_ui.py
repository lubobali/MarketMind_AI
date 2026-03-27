# Databricks notebook source
# MAGIC %md
# MAGIC # MarketMind AI — Phase 7: Gradio Chat UI
# MAGIC
# MAGIC **Module 6 coverage**: AI Agents + Interactive UI
# MAGIC
# MAGIC A professional chat interface and dashboard for the MarketMind AI agent.
# MAGIC
# MAGIC **Components:**
# MAGIC - **Chat tab**: Ask questions → agent calls tools → data-driven answers
# MAGIC - **Dashboard tab**: Sector heatmap, signal table, market overview (Plotly)
# MAGIC - Dark theme with MarketMind AI branding
# MAGIC - Example questions for quick start
# MAGIC
# MAGIC Run as a **regular notebook** on your cluster.

# COMMAND ----------

# MAGIC %pip install openai gradio plotly
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import json
import time

import gradio as gr
import plotly.graph_objects as go
from openai import OpenAI

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

CATALOG = "bootcamp_students"
SCHEMA = "lubo_marketmind_ai"
MODEL = "databricks-meta-llama-3-3-70b-instruct"

def table(name):
    return f"{CATALOG}.{SCHEMA}.{name}"

# ── OpenAI client pointed at Databricks Foundation Model API ──
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = OpenAI(
    api_key=token,
    base_url=f"https://{workspace_url}/serving-endpoints",
)

print(f"Model: {MODEL}")
print(f"Workspace: {workspace_url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Load Data from Delta Tables

# COMMAND ----------

def spark_to_dicts(table_name):
    """Read a Delta table and convert to list of dicts."""
    df = spark.table(table(table_name))
    rows = df.collect()
    return [row.asDict() for row in rows]

# Load all data
daily_records = spark_to_dicts("stock_daily_summary")
signal_records = spark_to_dicts("market_signals")
ranking_records = spark_to_dicts("sector_rankings")
moving_avg_records = spark_to_dicts("moving_averages")
spike_records = spark_to_dicts("volume_spikes")

# Sentiment tables
try:
    symbol_sentiments = spark_to_dicts("symbol_sentiment_agg")
    market_mood_rows = spark_to_dicts("market_mood")
    market_mood = market_mood_rows[0] if market_mood_rows else None
except Exception as e:
    print(f"Sentiment tables not available: {e}")
    symbol_sentiments = []
    market_mood = None

# Filter to latest date
latest_date = max(r["date"] for r in daily_records) if daily_records else None
latest_daily = [r for r in daily_records if r["date"] == latest_date]
latest_rankings = [r for r in ranking_records if r["date"] == latest_date]
latest_moving_avg = [r for r in moving_avg_records if r["date"] == latest_date]
latest_spikes = [r for r in spike_records if r["date"] == latest_date]

print(f"Data loaded — {len(latest_daily)} stocks, latest date: {latest_date}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Agent Tools + Loop (reused from Phase 6)

# COMMAND ----------

# ── Tool implementations ───────────────────────────────────────────────
# NOTE: These are inlined from utils/agent_tools.py because Databricks
# notebooks cannot import from local packages without `pip install -e .`.
# The CANONICAL source is utils/agent_tools.py (tested with 57 unit tests).
# If you modify tool logic, update utils/agent_tools.py FIRST, then sync here.

def get_stock_price(symbol):
    symbol = symbol.upper().strip()
    matches = [r for r in latest_daily if r["symbol"] == symbol]
    if not matches:
        return json.dumps({"error": f"No data found for {symbol}"})
    row = matches[0]
    return json.dumps({
        "symbol": row["symbol"], "date": str(row["date"]),
        "price": round(row["day_close"], 2), "open": round(row["day_open"], 2),
        "high": round(row["day_high"], 2), "low": round(row["day_low"], 2),
        "volume": row["total_volume"],
        "change_pct": round(row["avg_change_pct"], 2), "sector": row["sector"],
    })

def get_sector_performance(sector):
    sector = sector.strip().title()
    records = [r for r in latest_daily if r.get("sector") == sector]
    if not records:
        return json.dumps({"error": f"No data for sector '{sector}'"})
    rank_map = {r["symbol"]: r for r in latest_rankings if r.get("sector") == sector}
    stocks = []
    for rec in records:
        ri = rank_map.get(rec["symbol"], {})
        stocks.append({"symbol": rec["symbol"], "change_pct": round(rec["avg_change_pct"], 2),
                        "rank": ri.get("sector_rank"), "percentile": ri.get("percentile")})
    stocks.sort(key=lambda s: s["rank"] if s["rank"] is not None else 999)
    return json.dumps({"sector": sector, "date": str(records[0]["date"]),
                        "stock_count": len(stocks), "top_gainer": stocks[0]["symbol"],
                        "top_loser": stocks[-1]["symbol"], "stocks": stocks})

def get_market_sentiment(symbol=None):
    if symbol:
        symbol = symbol.upper().strip()
        match = next((r for r in symbol_sentiments if r["symbol"] == symbol), None)
        if not match:
            return json.dumps({"error": f"No sentiment data for {symbol}"})
        score = match["avg_sentiment"]
        label = "bullish" if score > 0.05 else "bearish" if score < -0.05 else "neutral"
        return json.dumps({"symbol": match["symbol"], "avg_sentiment": round(score, 3),
                            "label": label, "article_count": match["article_count"],
                            "most_positive": match["most_positive"], "most_negative": match["most_negative"]})
    if not market_mood:
        return json.dumps({"error": "No market mood data available"})
    return json.dumps({"mood": market_mood["mood"], "avg_score": round(market_mood["avg_score"], 3),
                        "article_count": market_mood["article_count"],
                        "positive_count": market_mood["positive_count"],
                        "negative_count": market_mood["negative_count"],
                        "neutral_count": market_mood["neutral_count"]})

def get_technical_signals(symbol):
    symbol = symbol.upper().strip()
    match = next((r for r in signal_records if r["symbol"] == symbol), None)
    if not match:
        return json.dumps({"error": f"No technical data for {symbol}"})
    return json.dumps({
        "symbol": match["symbol"], "sector": match["sector"],
        "date": str(match["latest_date"]), "price": round(match["latest_close"], 2),
        "rsi": round(match["rsi"], 2) if match["rsi"] else None,
        "macd": {"line": round(match["macd_line"], 4) if match["macd_line"] else None,
                 "signal": round(match["macd_signal"], 4) if match["macd_signal"] else None,
                 "histogram": round(match["macd_histogram"], 4) if match["macd_histogram"] else None},
        "bollinger": {"upper": round(match["bb_upper"], 2) if match["bb_upper"] else None,
                      "middle": round(match["bb_middle"], 2) if match["bb_middle"] else None,
                      "lower": round(match["bb_lower"], 2) if match["bb_lower"] else None},
        "signal": match["signal"]})

def compare_stocks(symbol1, symbol2):
    symbol1, symbol2 = symbol1.upper().strip(), symbol2.upper().strip()
    avg_map = {r["symbol"]: r for r in latest_moving_avg}
    stocks = {}
    for sym in (symbol1, symbol2):
        sig = next((r for r in signal_records if r["symbol"] == sym), None)
        if not sig:
            return json.dumps({"error": f"Could not find data for {sym}"})
        avg = avg_map.get(sym, {})
        stocks[sym] = {"sector": sig["sector"], "price": round(sig["latest_close"], 2),
                        "rsi": round(sig["rsi"], 2) if sig.get("rsi") else None,
                        "macd_histogram": round(sig["macd_histogram"], 4) if sig.get("macd_histogram") else None,
                        "signal": sig["signal"],
                        "sma_5": round(avg["sma_5"], 2) if avg.get("sma_5") else None,
                        "sma_20": round(avg["sma_20"], 2) if avg.get("sma_20") else None,
                        "vwap_20": round(avg["vwap_20"], 2) if avg.get("vwap_20") else None}
    return json.dumps({"comparison": stocks})

def get_market_summary():
    if not latest_daily:
        return json.dumps({"error": "No daily data available"})
    by_change = sorted(latest_daily, key=lambda r: r["avg_change_pct"], reverse=True)
    by_vol = sorted(latest_daily, key=lambda r: r["total_volume"], reverse=True)
    sig_counts = {}
    for r in signal_records:
        sig_counts[r["signal"]] = sig_counts.get(r["signal"], 0) + 1
    spikes = [{"symbol": r["symbol"], "ratio": round(r["volume_ratio"], 2)}
              for r in latest_spikes if r.get("is_spike")]
    return json.dumps({
        "date": str(latest_date), "total_stocks": len(latest_daily),
        "top_gainers": [{"symbol": r["symbol"], "change_pct": round(r["avg_change_pct"], 2)} for r in by_change[:3]],
        "top_losers": [{"symbol": r["symbol"], "change_pct": round(r["avg_change_pct"], 2)} for r in by_change[-3:][::-1]],
        "most_active": [{"symbol": r["symbol"], "volume": r["total_volume"]} for r in by_vol[:3]],
        "signal_distribution": sig_counts, "volume_spikes_today": spikes})

TOOL_REGISTRY = {
    "get_stock_price": lambda **kw: get_stock_price(**kw),
    "get_sector_performance": lambda **kw: get_sector_performance(**kw),
    "get_market_sentiment": lambda **kw: get_market_sentiment(**kw),
    "get_technical_signals": lambda **kw: get_technical_signals(**kw),
    "compare_stocks": lambda **kw: compare_stocks(**kw),
    "get_market_summary": lambda **kw: get_market_summary(**kw),
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_stock_price",
        "description": "Get the latest price, daily change, and volume for a stock symbol.",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "Stock ticker, e.g. AAPL"}},
            "required": ["symbol"]}}},
    {"type": "function", "function": {
        "name": "get_sector_performance",
        "description": "Get performance for a sector. Sectors: Technology, Finance, Energy, Healthcare, Consumer.",
        "parameters": {"type": "object", "properties": {
            "sector": {"type": "string", "description": "Sector name"}},
            "required": ["sector"]}}},
    {"type": "function", "function": {
        "name": "get_market_sentiment",
        "description": "Get news sentiment for a stock or overall market mood. Omit symbol for market-wide.",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "Optional stock ticker"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "get_technical_signals",
        "description": "Get RSI, MACD, Bollinger Bands, and buy/sell/hold signal for a stock.",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "Stock ticker"}},
            "required": ["symbol"]}}},
    {"type": "function", "function": {
        "name": "compare_stocks",
        "description": "Compare two stocks side-by-side — price, technicals, and signal.",
        "parameters": {"type": "object", "properties": {
            "symbol1": {"type": "string", "description": "First ticker"},
            "symbol2": {"type": "string", "description": "Second ticker"}},
            "required": ["symbol1", "symbol2"]}}},
    {"type": "function", "function": {
        "name": "get_market_summary",
        "description": "Overall market snapshot — top gainers, losers, signals, volume spikes.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]

SYSTEM_PROMPT = """You are MarketMind AI, an intelligent stock market analyst powered by real-time data.

Use your tools to query live market data and provide accurate, data-driven answers about stock
prices, technical analysis, sector performance, news sentiment, and market trends.

Rules:
- Always use tools to get data — never make up numbers
- Explain technical indicators in plain language
- Give clear opinions when asked, backed by data
- Format responses with markdown for readability
- Be concise but thorough

You cover 15 stocks across 5 sectors: Technology (AAPL, MSFT, NVDA, GOOGL, AMZN),
Finance (JPM, GS, BAC), Energy (XOM, CVX), Healthcare (PFE, JNJ), Consumer (TSLA, WMT, KO)."""

print(f"Registered {len(TOOL_REGISTRY)} tools")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3: Agent Loop

# COMMAND ----------

def run_agent(query, max_iterations=5):
    """Run MarketMind AI agent with tool calling."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    tools_used = []
    start = time.time()

    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOL_SCHEMAS,
            tool_choice="auto", temperature=0.1, max_tokens=2000,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            elapsed = round(time.time() - start, 2)
            return {"answer": msg.content, "tools_used": tools_used,
                    "iterations": iteration + 1, "latency_seconds": elapsed}

        messages.append({
            "role": "assistant", "content": msg.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                           for tc in msg.tool_calls],
        })

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                fn_args = {}
                result = json.dumps({"error": f"Malformed arguments for {fn_name}"})
                tools_used.append({"tool": fn_name, "args": fn_args})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                continue
            fn = TOOL_REGISTRY.get(fn_name)
            result = fn(**fn_args) if fn else json.dumps({"error": f"Unknown tool: {fn_name}"})
            tools_used.append({"tool": fn_name, "args": fn_args})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    elapsed = round(time.time() - start, 2)
    return {"answer": "Agent reached max iterations.", "tools_used": tools_used,
            "iterations": max_iterations, "latency_seconds": elapsed}

# Quick test
test_result = run_agent("What is NVDA trading at?")
print(f"Agent test: {test_result['latency_seconds']}s, {len(test_result['tools_used'])} tools")
print(f"Answer: {test_result['answer'][:200]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4: Plotly Charts for Dashboard

# COMMAND ----------

# ── Color palette ────────────────────────────────────────────

COLORS = {
    "bg": "#0d1117",
    "card": "#161b22",
    "text": "#e6edf3",
    "muted": "#8b949e",
    "green": "#3fb950",
    "red": "#f85149",
    "blue": "#58a6ff",
    "yellow": "#d29922",
    "border": "#30363d",
}

DARK_LAYOUT = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["card"],
    font=dict(color=COLORS["text"], family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=50, b=40),
)


def build_sector_heatmap():
    """Bar chart of stock performance colored by change direction."""
    sorted_records = sorted(latest_daily, key=lambda r: r["avg_change_pct"], reverse=True)
    symbols = [r["symbol"] for r in sorted_records]
    changes = [r["avg_change_pct"] for r in sorted_records]
    colors = [COLORS["green"] if c >= 0 else COLORS["red"] for c in changes]

    fig = go.Figure(data=[go.Bar(
        x=symbols, y=changes, marker_color=colors,
        text=[f"{c:+.2f}%" for c in changes], textposition="outside",
        hovertemplate="<b>%{x}</b><br>Change: %{y:.2f}%<extra></extra>",
    )])
    fig.update_layout(
        title=dict(text="Sector Performance — Daily Change %", font=dict(size=16)),
        yaxis_title="Change %", showlegend=False, height=400, **DARK_LAYOUT,
    )
    return fig


def build_signal_table():
    """Plotly table with buy/sell/hold signals and technicals."""
    headers = ["Symbol", "Sector", "Price", "RSI", "MACD Hist", "Signal"]
    sorted_sigs = sorted(signal_records, key=lambda r: r["symbol"])

    symbols = [r["symbol"] for r in sorted_sigs]
    sectors = [r["sector"] for r in sorted_sigs]
    prices = [f"${r['latest_close']:.2f}" for r in sorted_sigs]
    rsis = [f"{r['rsi']:.1f}" if r.get("rsi") else "—" for r in sorted_sigs]
    macds = [f"{r['macd_histogram']:.4f}" if r.get("macd_histogram") else "—" for r in sorted_sigs]
    signals = [r["signal"].upper() for r in sorted_sigs]

    signal_colors = []
    for s in signals:
        if s == "BUY":
            signal_colors.append(COLORS["green"])
        elif s == "SELL":
            signal_colors.append(COLORS["red"])
        else:
            signal_colors.append(COLORS["yellow"])

    fig = go.Figure(data=[go.Table(
        header=dict(values=headers, fill_color=COLORS["card"],
                    font=dict(color=COLORS["text"], size=13),
                    line_color=COLORS["border"], align="left"),
        cells=dict(values=[symbols, sectors, prices, rsis, macds, signals],
                   fill_color=COLORS["bg"],
                   font=dict(color=[COLORS["text"]] * 5 + [signal_colors], size=12),
                   line_color=COLORS["border"], align="left"),
    )])
    fig.update_layout(
        title=dict(text="Technical Signals — All Stocks", font=dict(size=16)),
        height=450, **DARK_LAYOUT,
    )
    return fig


def build_market_overview():
    """Horizontal bar chart sorted by daily change %."""
    sorted_records = sorted(latest_daily, key=lambda r: r["avg_change_pct"])
    symbols = [r["symbol"] for r in sorted_records]
    changes = [r["avg_change_pct"] for r in sorted_records]
    colors = [COLORS["green"] if c >= 0 else COLORS["red"] for c in changes]

    fig = go.Figure(data=[go.Bar(
        x=changes, y=symbols, orientation="h", marker_color=colors,
        text=[f"{c:+.2f}%" for c in changes], textposition="outside",
        hovertemplate="<b>%{y}</b><br>Change: %{x:.2f}%<extra></extra>",
    )])
    fig.update_layout(
        title=dict(text="Market Overview — Daily Performance", font=dict(size=16)),
        xaxis_title="Change %", showlegend=False, height=500, **DARK_LAYOUT,
    )
    return fig


# Preview charts
sector_fig = build_sector_heatmap()
signal_fig = build_signal_table()
overview_fig = build_market_overview()
print("Charts built successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 5: Gradio App

# COMMAND ----------

# ── Custom CSS for MarketMind branding ──────────────────────

CUSTOM_CSS = """
.gradio-container {
    max-width: 1200px !important;
    font-family: 'Inter', sans-serif !important;
}
.main-header {
    text-align: center;
    padding: 20px 0 10px 0;
}
.main-header h1 {
    font-size: 2em;
    background: linear-gradient(135deg, #58a6ff, #3fb950);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}
.main-header p {
    color: #8b949e;
    font-size: 1.1em;
}
footer { display: none !important; }
"""

# ── Chat handler ────────────────────────────────────────────

def chat_fn(message, history):
    """Handle chat messages — run the agent and return answer."""
    if not message.strip():
        return "Please ask a question about the stock market!"
    try:
        result = run_agent(message)
        tools = ", ".join(t["tool"] for t in result["tools_used"])
        footer = f"\n\n---\n*{result['latency_seconds']}s · {len(result['tools_used'])} tool calls · {result['iterations']} iterations*"
        return result["answer"] + footer
    except Exception as e:
        return f"Sorry, an error occurred: {e}"

# ── Example questions ───────────────────────────────────────

EXAMPLES = [
    "What's the market mood today?",
    "Compare AAPL vs MSFT — which is a better buy?",
    "Which stocks have bullish signals right now?",
    "How is the Technology sector performing?",
    "What are the technical signals for NVDA?",
    "Give me an overall market summary",
    "Why might TSLA be a buy signal?",
    "What is JPM trading at today?",
]

# ── Build the Gradio app ───────────────────────────────────

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="blue",
        neutral_hue="gray",
    ).set(
        body_background_fill="#0d1117",
        body_background_fill_dark="#0d1117",
        block_background_fill="#161b22",
        block_background_fill_dark="#161b22",
        block_border_color="#30363d",
        input_background_fill="#0d1117",
        input_background_fill_dark="#0d1117",
    ),
    css=CUSTOM_CSS,
    title="MarketMind AI",
) as app:

    # ── Header ──
    gr.HTML("""
    <div class="main-header">
        <h1>🧠 MarketMind AI</h1>
        <p>AI-Powered Stock Market Analyst · 15 Stocks · 5 Sectors · Real-Time Data</p>
    </div>
    """)

    with gr.Tabs():
        # ══════════════ Tab 1: Chat ══════════════
        with gr.TabItem("💬 Chat", id="chat"):
            chatbot = gr.ChatInterface(
                fn=chat_fn,
                examples=EXAMPLES,
                title="",
                description="Ask me anything about stocks, sectors, technicals, or market sentiment.",
            )

        # ══════════════ Tab 2: Dashboard ══════════════
        with gr.TabItem("📊 Dashboard", id="dashboard"):
            gr.Markdown(f"### Market Data — {latest_date}")

            with gr.Row():
                with gr.Column(scale=1):
                    # Summary stats
                    sig_counts = {}
                    for r in signal_records:
                        sig_counts[r["signal"]] = sig_counts.get(r["signal"], 0) + 1

                    buy_count = sig_counts.get("buy", 0)
                    sell_count = sig_counts.get("sell", 0)
                    hold_count = sig_counts.get("hold", 0)

                    gr.Markdown(f"""
**{len(latest_daily)} Stocks Tracked** across 5 sectors

| Signal | Count |
|--------|-------|
| 🟢 Buy | {buy_count} |
| 🔴 Sell | {sell_count} |
| 🟡 Hold | {hold_count} |
""")

            with gr.Row():
                gr.Plot(value=sector_fig, label="Sector Performance")

            with gr.Row():
                gr.Plot(value=overview_fig, label="Market Overview")

        # ══════════════ Tab 3: Signals ══════════════
        with gr.TabItem("📈 Signals", id="signals"):
            gr.Markdown("### Technical Signals — Buy / Sell / Hold")
            gr.Plot(value=signal_fig, label="Signal Table")

            gr.Markdown("""
**How to read signals:**
- **RSI > 70** = Overbought (potential sell)
- **RSI < 30** = Oversold (potential buy)
- **MACD Histogram > 0** = Bullish momentum
- **MACD Histogram < 0** = Bearish momentum
- **Signal** = Combined score from RSI + MACD + Bollinger Bands
""")

    gr.Markdown("""
---
*MarketMind AI · Databricks + Spark Bootcamp Capstone · Powered by Meta Llama 3.3 70B*
""")

print("Gradio app built — ready to launch")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 6: Launch the App
# MAGIC
# MAGIC Running `app.launch()` starts a Gradio server. In Databricks, this creates
# MAGIC a proxy URL accessible from your browser.

# COMMAND ----------

# Launch Gradio — creates a shareable URL in Databricks
app.launch(share=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Component | Details |
# MAGIC |-----------|---------|
# MAGIC | **Chat** | AI agent with 6 tools, Llama 3.3 70B, multi-step reasoning |
# MAGIC | **Dashboard** | Sector heatmap + market overview (Plotly, dark theme) |
# MAGIC | **Signals** | Technical signal table with RSI, MACD, buy/sell/hold |
# MAGIC | **Theme** | Dark finance aesthetic, MarketMind AI branding |
# MAGIC | **Examples** | 8 quick-start questions for new users |
# MAGIC | **Data** | 15 stocks, 5 sectors, 7 Delta Lake tables |
