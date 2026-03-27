# Databricks notebook source
# MAGIC %md
# MAGIC # MarketMind AI — Phase 6: AI Agent
# MAGIC
# MAGIC **Module 6 coverage**: Building AI Agents with Databricks
# MAGIC
# MAGIC An AI financial analyst that uses **tool calling** to query Delta tables and
# MAGIC answer natural language questions about stocks, sectors, sentiment, and signals.
# MAGIC
# MAGIC **Components:**
# MAGIC - 6 tools querying `bootcamp_students.lubo_marketmind_ai` tables
# MAGIC - Databricks Foundation Model (Meta Llama 3.3 70B) with function calling
# MAGIC - Multi-step agent loop with tool dispatch
# MAGIC - MLflow experiment tracking + model logging
# MAGIC
# MAGIC Run as a **regular notebook** on your cluster.

# COMMAND ----------

# MAGIC %pip install openai mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import json
import os
import time

import mlflow
import pandas as pd
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
# MAGIC
# MAGIC Pre-query all tables into Python lists so the agent tools can work
# MAGIC without Spark dependencies inside the loop.

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

# Sentiment tables (from DLT pipeline)
try:
    symbol_sentiments = spark_to_dicts("symbol_sentiment_agg")
    market_mood_rows = spark_to_dicts("market_mood")
    market_mood = market_mood_rows[0] if market_mood_rows else None
except Exception as e:
    print(f"Sentiment tables not available: {e}")
    symbol_sentiments = []
    market_mood = None

# Filter to latest date for daily records
latest_date = max(r["date"] for r in daily_records) if daily_records else None
latest_daily = [r for r in daily_records if r["date"] == latest_date]
latest_rankings = [r for r in ranking_records if r["date"] == latest_date]
latest_moving_avg = [r for r in moving_avg_records if r["date"] == latest_date]
latest_spikes = [r for r in spike_records if r["date"] == latest_date]

print(f"Latest date: {latest_date}")
print(f"Daily records (latest): {len(latest_daily)}")
print(f"Signal records: {len(signal_records)}")
print(f"Rankings (latest): {len(latest_rankings)}")
print(f"Moving averages (latest): {len(latest_moving_avg)}")
print(f"Volume spikes (latest): {len(latest_spikes)}")
print(f"Symbol sentiments: {len(symbol_sentiments)}")
print(f"Market mood: {'available' if market_mood else 'not available'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Agent Tools + Loop
# MAGIC
# MAGIC Pure-Python tools that format data. Agent loop dispatches them via LLM tool calling.

# COMMAND ----------

# ── Tool implementations (inline — same logic as utils/agent_tools.py) ──

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

# COMMAND ----------

# ── Tool registry and schemas ────────────────────────────────

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
- Be concise but thorough

You cover 15 stocks across 5 sectors: Technology (AAPL, MSFT, NVDA, GOOGL, AMZN),
Finance (JPM, GS, BAC), Energy (XOM, CVX), Healthcare (PFE, JNJ), Consumer (TSLA, WMT, KO)."""

print(f"Registered {len(TOOL_REGISTRY)} tools: {list(TOOL_REGISTRY.keys())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 3: Agent Loop

# COMMAND ----------

def run_agent(query, max_iterations=5, verbose=True):
    """Run MarketMind AI agent with tool calling."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    tools_used = []
    start = time.time()

    for iteration in range(max_iterations):
        if verbose:
            print(f"\n--- Iteration {iteration + 1} ---")

        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOL_SCHEMAS,
            tool_choice="auto", temperature=0.1, max_tokens=2000,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            elapsed = round(time.time() - start, 2)
            if verbose:
                print(f"\nFinal answer ({elapsed}s, {len(tools_used)} tool calls):")
                print(msg.content)
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
            fn_args = json.loads(tc.function.arguments)
            if verbose:
                print(f"  Tool: {fn_name}({fn_args})")

            fn = TOOL_REGISTRY.get(fn_name)
            result = fn(**fn_args) if fn else json.dumps({"error": f"Unknown tool: {fn_name}"})
            tools_used.append({"tool": fn_name, "args": fn_args})

            if verbose:
                print(f"  Result: {result[:200]}...")

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    elapsed = round(time.time() - start, 2)
    return {"answer": "Agent reached max iterations.", "tools_used": tools_used,
            "iterations": max_iterations, "latency_seconds": elapsed}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4: Test the Agent
# MAGIC
# MAGIC Run example queries to verify tool calling works end-to-end.

# COMMAND ----------

# Test 1: Simple stock price
result1 = run_agent("What is NVDA trading at today?")

# COMMAND ----------

# Test 2: Technical analysis + recommendation
result2 = run_agent("What are the technical signals for MSFT? Should I buy or sell?")

# COMMAND ----------

# Test 3: Stock comparison (multi-tool)
result3 = run_agent("Compare AAPL and GOOGL — which is a better buy right now?")

# COMMAND ----------

# Test 4: Sector analysis
result4 = run_agent("How is the Technology sector performing? Who are the winners and losers?")

# COMMAND ----------

# Test 5: Market overview
result5 = run_agent("Give me an overall market summary. What stocks have bullish signals?")

# COMMAND ----------

# Test 6: Multi-step reasoning
result6 = run_agent("Why might TSLA be a buy signal despite its recent price drop?")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 5: MLflow Experiment Tracking

# COMMAND ----------

EXPERIMENT_NAME = "/Users/data@lubobali.com/MarketMind_AI_Agent"
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"MLflow experiment: {EXPERIMENT_NAME}")

# COMMAND ----------

# ── Log agent runs ───────────────────────────────────────────

TEST_QUERIES = [
    "What is NVDA trading at today?",
    "What are the technical signals for MSFT? Should I buy or sell?",
    "Compare AAPL and GOOGL — which is a better buy?",
    "How is the Technology sector performing?",
    "Give me an overall market summary.",
    "Why might TSLA be a buy signal despite its recent drop?",
]

all_results = []

for i, query in enumerate(TEST_QUERIES):
    with mlflow.start_run(run_name=f"query_{i+1}"):
        mlflow.log_param("model", MODEL)
        mlflow.log_param("query", query)
        mlflow.log_param("max_iterations", 5)

        result = run_agent(query, verbose=False)
        all_results.append(result)

        mlflow.log_metric("latency_seconds", result["latency_seconds"])
        mlflow.log_metric("tool_calls", len(result["tools_used"]))
        mlflow.log_metric("iterations", result["iterations"])
        mlflow.log_metric("answer_length", len(result["answer"]) if result["answer"] else 0)

        with open(f"/tmp/agent_result_{i+1}.json", "w") as f:
            json.dump({"query": query, **result}, f, indent=2)
        mlflow.log_artifact(f"/tmp/agent_result_{i+1}.json")

        tools_str = ", ".join([t["tool"] for t in result["tools_used"]])
        mlflow.set_tag("tools_used", tools_str)

        print(f"  Query {i+1}: {len(result['tools_used'])} tools, {result['latency_seconds']}s")

print(f"\nLogged {len(TEST_QUERIES)} runs to {EXPERIMENT_NAME}")

# COMMAND ----------

# ── Performance summary ──────────────────────────────────────

print("=" * 70)
print("AGENT PERFORMANCE SUMMARY")
print("=" * 70)

total_latency = sum(r["latency_seconds"] for r in all_results)
total_tools = sum(len(r["tools_used"]) for r in all_results)

for i, (query, result) in enumerate(zip(TEST_QUERIES, all_results)):
    print(f"\nQuery {i+1}: {query}")
    print(f"  Tools: {[t['tool'] for t in result['tools_used']]}")
    print(f"  Iterations: {result['iterations']}, Latency: {result['latency_seconds']}s")
    print(f"  Answer: {result['answer'][:150]}...")

print(f"\n{'=' * 70}")
print(f"Total queries: {len(TEST_QUERIES)}")
print(f"Total tool calls: {total_tools}")
print(f"Avg latency: {round(total_latency / len(TEST_QUERIES), 2)}s")
print(f"Avg tools/query: {round(total_tools / len(TEST_QUERIES), 1)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 6: Log Agent as MLflow Model

# COMMAND ----------

class MarketMindAgentModel(mlflow.pyfunc.PythonModel):
    """MLflow-packaged MarketMind AI Agent."""

    def load_context(self, context):
        self.model_name = MODEL

    def predict(self, context, model_input):
        queries = model_input["query"].tolist()
        return [run_agent(q, verbose=False)["answer"] for q in queries]

# COMMAND ----------

with mlflow.start_run(run_name="marketmind_agent_model") as run:
    mlflow.log_param("model", MODEL)
    mlflow.log_param("num_tools", len(TOOL_REGISTRY))
    mlflow.log_param("tools", list(TOOL_REGISTRY.keys()))

    input_example = pd.DataFrame({"query": ["What is NVDA trading at?"]})

    model_info = mlflow.pyfunc.log_model(
        artifact_path="marketmind_agent",
        python_model=MarketMindAgentModel(),
        input_example=input_example,
        pip_requirements=["openai", "mlflow"],
    )
    print(f"Model logged: {model_info.model_uri}")
    print(f"Run ID: {run.info.run_id}")

# COMMAND ----------

# ── Register in Model Registry ───────────────────────────────

MODEL_REGISTRY_NAME = "MarketMind_AI_Agent"

try:
    model_version = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=MODEL_REGISTRY_NAME,
    )
    print(f"Registered: {MODEL_REGISTRY_NAME} v{model_version.version}")
except Exception as e:
    print(f"Registry note: {e}")
    print("(Model was still logged to MLflow. Registry may need UC permissions.)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Component | Details |
# MAGIC |-----------|---------|
# MAGIC | **LLM** | Meta Llama 3.3 70B Instruct (Databricks Foundation Model) |
# MAGIC | **Tools** | 6 tools querying 7 Delta Lake tables |
# MAGIC | **Agent loop** | Multi-step reasoning, max 5 iterations |
# MAGIC | **MLflow** | 6 runs logged + model registered |
# MAGIC | **Tables** | stock_daily_summary, market_signals, moving_averages, sector_rankings, volume_spikes, symbol_sentiment_agg, market_mood |

# COMMAND ----------

print("MarketMind AI Agent — Phase 6 Complete")
print(f"  Model: {MODEL}")
print(f"  Tools: {len(TOOL_REGISTRY)}")
print(f"  MLflow: {EXPERIMENT_NAME}")
print(f"  Queries tested: {len(TEST_QUERIES)}")
