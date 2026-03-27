"""Tests for utils/agent.py — Agent loop with tool calling.

We mock the LLM and test the pipeline: does the agent correctly
dispatch tool calls, pass results back, and return the final answer?
"""

from unittest.mock import MagicMock

import pytest

from utils.agent import TOOL_SCHEMAS, MarketMindAgent

# ── Helpers for building mock LLM responses ──────────────────


def _make_final_response(content: str):
    """Build a mock LLM response with no tool calls (final answer)."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_tool_call_response(tool_name: str, arguments: str, call_id: str = "call_1"):
    """Build a mock LLM response with a single tool call."""
    tool_call = MagicMock()
    tool_call.id = call_id
    tool_call.function.name = tool_name
    tool_call.function.arguments = arguments

    msg = MagicMock()
    msg.content = ""
    msg.tool_calls = [tool_call]

    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


# ════════════════════════════════════════════════════════════════
#  Tool Schema Tests
# ════════════════════════════════════════════════════════════════


class TestToolSchemas:
    """Verify tool schemas are well-formed."""

    def test_six_tools_defined(self):
        assert len(TOOL_SCHEMAS) == 6

    def test_all_have_required_fields(self):
        for schema in TOOL_SCHEMAS:
            assert schema["type"] == "function"
            assert "name" in schema["function"]
            assert "description" in schema["function"]
            assert "parameters" in schema["function"]

    @pytest.mark.parametrize(
        "tool_name",
        [
            "get_stock_price",
            "get_sector_performance",
            "get_market_sentiment",
            "get_technical_signals",
            "compare_stocks",
            "get_market_summary",
        ],
    )
    def test_tool_exists(self, tool_name):
        names = [s["function"]["name"] for s in TOOL_SCHEMAS]
        assert tool_name in names


# ════════════════════════════════════════════════════════════════
#  Agent Loop Tests
# ════════════════════════════════════════════════════════════════


class TestMarketMindAgent:
    """Tests for the agent loop — mock LLM, test pipeline."""

    @pytest.fixture
    def mock_data(self):
        """Minimal data store for the agent."""
        return {
            "daily_records": [
                {
                    "symbol": "NVDA",
                    "date": "2026-03-26",
                    "sector": "Technology",
                    "day_open": 173.50,
                    "day_high": 176.80,
                    "day_low": 170.10,
                    "day_close": 171.24,
                    "total_volume": 182_162_282,
                    "avg_change_pct": -2.05,
                },
            ],
            "signal_records": [
                {
                    "symbol": "NVDA",
                    "sector": "Technology",
                    "latest_date": "2026-03-26",
                    "latest_close": 171.24,
                    "rsi": 38.76,
                    "macd_line": -2.83,
                    "macd_signal": -2.04,
                    "macd_histogram": -0.79,
                    "bb_upper": 187.72,
                    "bb_middle": 179.91,
                    "bb_lower": 172.10,
                    "signal": "buy",
                },
            ],
            "ranking_records": [],
            "moving_avg_records": [],
            "symbol_sentiments": [],
            "market_mood": None,
            "spike_records": [],
        }

    @pytest.fixture
    def agent(self, mock_data):
        """Agent with a mocked LLM client."""
        mock_client = MagicMock()
        return MarketMindAgent(client=mock_client, model="test-model", data=mock_data)

    def test_direct_answer_no_tools(self, agent):
        """If LLM returns an answer without tool calls, agent returns it."""
        agent.client.chat.completions.create.return_value = _make_final_response("NVDA is at $171.24")

        result = agent.run("What is NVDA at?")
        assert result["answer"] == "NVDA is at $171.24"
        assert result["tools_used"] == []
        assert result["iterations"] == 1

    def test_single_tool_call(self, agent):
        """Agent calls one tool, passes result to LLM, gets final answer."""
        # First call: LLM requests a tool
        tool_response = _make_tool_call_response("get_stock_price", '{"symbol": "NVDA"}')
        # Second call: LLM gives final answer
        final_response = _make_final_response("NVDA is trading at $171.24, down 2.05%.")

        agent.client.chat.completions.create.side_effect = [tool_response, final_response]

        result = agent.run("What is NVDA at?")
        assert "171.24" in result["answer"]
        assert len(result["tools_used"]) == 1
        assert result["tools_used"][0]["tool"] == "get_stock_price"
        assert result["iterations"] == 2

    def test_tool_result_passed_back_to_llm(self, agent):
        """The tool's output should be passed back to the LLM as a tool message."""
        tool_response = _make_tool_call_response("get_stock_price", '{"symbol": "NVDA"}')
        final_response = _make_final_response("Done.")

        agent.client.chat.completions.create.side_effect = [tool_response, final_response]
        agent.run("Price of NVDA?")

        # Second call should include the tool result in messages
        second_call_messages = agent.client.chat.completions.create.call_args_list[1][1]["messages"]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert "171.24" in tool_messages[0]["content"]

    def test_max_iterations_safety(self, agent):
        """Agent should stop after max_iterations even if LLM keeps calling tools."""
        # LLM always requests tools, never gives final answer
        tool_response = _make_tool_call_response("get_stock_price", '{"symbol": "NVDA"}')
        agent.client.chat.completions.create.return_value = tool_response

        result = agent.run("Price?", max_iterations=3)
        assert result["iterations"] == 3
        assert "max iterations" in result["answer"].lower()

    def test_unknown_tool_handled_gracefully(self, agent):
        """If LLM calls a tool that doesn't exist, agent handles it."""
        tool_response = _make_tool_call_response("nonexistent_tool", "{}")
        final_response = _make_final_response("Sorry, I couldn't find that.")

        agent.client.chat.completions.create.side_effect = [tool_response, final_response]

        result = agent.run("Do something weird")
        # Should not crash — unknown tool returns error string
        assert result["answer"] is not None

    def test_malformed_json_arguments_handled(self, agent):
        """If LLM emits malformed JSON arguments, agent handles gracefully."""
        tool_response = _make_tool_call_response("get_stock_price", "not valid json{{{")
        final_response = _make_final_response("Sorry, there was an error.")

        agent.client.chat.completions.create.side_effect = [tool_response, final_response]

        result = agent.run("Price of NVDA?")
        assert result["answer"] is not None
        assert len(result["tools_used"]) == 1
        assert result["tools_used"][0]["args"] == {}

    def test_result_includes_latency(self, agent):
        """Result dict should include latency_seconds."""
        agent.client.chat.completions.create.return_value = _make_final_response("Done.")
        result = agent.run("Hi")
        assert "latency_seconds" in result
        assert isinstance(result["latency_seconds"], float)
