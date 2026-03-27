"""MarketMind AI — Agent loop with tool calling.

Orchestrates LLM calls and tool dispatch. The LLM decides which tools
to call, the agent executes them and passes results back.

Designed to work with any OpenAI-compatible client (Databricks Foundation
Model API, OpenAI, etc.).
"""

import json
import time

from utils.agent_tools import (
    compare_stocks,
    get_market_sentiment,
    get_market_summary,
    get_sector_performance,
    get_stock_price,
    get_technical_signals,
)

# ── Tool schemas (OpenAI function calling format) ────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the latest price, daily change, and volume for a stock symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker, e.g. AAPL, NVDA"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sector_performance",
            "description": "Get performance metrics for a market sector. Sectors: Technology, Finance, Energy, Healthcare, Consumer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sector": {"type": "string", "description": "Sector name"},
                },
                "required": ["sector"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_sentiment",
            "description": "Get news sentiment for a stock or overall market mood. Omit symbol for market-wide sentiment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Optional stock ticker"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_technical_signals",
            "description": "Get RSI, MACD, Bollinger Bands, and buy/sell/hold signal for a stock.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_stocks",
            "description": "Compare two stocks side-by-side — price, technicals, moving averages, and signal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol1": {"type": "string", "description": "First stock ticker"},
                    "symbol2": {"type": "string", "description": "Second stock ticker"},
                },
                "required": ["symbol1", "symbol2"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_summary",
            "description": "Get an overall market snapshot — top gainers, losers, most active, signals, volume spikes.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
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


class MarketMindAgent:
    """Agent that combines LLM reasoning with tool calling.

    Args:
        client: OpenAI-compatible client instance.
        model: Model name/endpoint.
        data: Dict with pre-queried data for tools:
            daily_records, signal_records, ranking_records,
            moving_avg_records, symbol_sentiments, market_mood, spike_records.
    """

    def __init__(self, client, model: str, data: dict):
        self.client = client
        self.model = model
        self.data = data

    def _dispatch_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool and return JSON string result."""
        d = self.data

        if tool_name == "get_stock_price":
            result = get_stock_price(d["daily_records"], arguments["symbol"])
        elif tool_name == "get_sector_performance":
            result = get_sector_performance(d["daily_records"], d["ranking_records"], arguments["sector"])
        elif tool_name == "get_market_sentiment":
            result = get_market_sentiment(d["symbol_sentiments"], d["market_mood"], arguments.get("symbol"))
        elif tool_name == "get_technical_signals":
            result = get_technical_signals(d["signal_records"], arguments["symbol"])
        elif tool_name == "compare_stocks":
            result = compare_stocks(
                d["signal_records"],
                d["moving_avg_records"],
                arguments["symbol1"],
                arguments["symbol2"],
            )
        elif tool_name == "get_market_summary":
            result = get_market_summary(d["daily_records"], d["signal_records"], d["spike_records"])
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        return json.dumps(result)

    def run(self, query: str, max_iterations: int = 5) -> dict:
        """Run the agent on a user query.

        Args:
            query: Natural language question.
            max_iterations: Max tool-calling rounds.

        Returns:
            Dict with answer, tools_used, iterations, latency_seconds.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        tools_used = []
        start_time = time.time()

        for iteration in range(max_iterations):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=2000,
            )

            assistant_msg = response.choices[0].message

            # No tool calls → final answer
            if not assistant_msg.tool_calls:
                return {
                    "answer": assistant_msg.content,
                    "tools_used": tools_used,
                    "iterations": iteration + 1,
                    "latency_seconds": round(time.time() - start_time, 2),
                }

            # Append assistant message with tool calls
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in assistant_msg.tool_calls
                    ],
                }
            )

            # Execute each tool call
            for tc in assistant_msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    fn_args = {}
                    result_str = json.dumps({"error": f"Malformed arguments for {fn_name}"})
                    tools_used.append({"tool": fn_name, "args": fn_args})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})
                    continue

                result_str = self._dispatch_tool(fn_name, fn_args)
                tools_used.append({"tool": fn_name, "args": fn_args})

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    }
                )

        return {
            "answer": "Agent reached max iterations without a final answer.",
            "tools_used": tools_used,
            "iterations": max_iterations,
            "latency_seconds": round(time.time() - start_time, 2),
        }
