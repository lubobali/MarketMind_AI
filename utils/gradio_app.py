"""MarketMind AI — Gradio UI utilities.

Testable chart builders and chat handler. Pure Python — no Spark dependency.
Takes pre-queried data (lists of dicts) and returns Plotly figures or strings.
The Databricks notebook handles data loading and Gradio wiring.
"""

import plotly.graph_objects as go

# ── Color palette (dark finance theme) ──────────────────────────

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


# ════════════════════════════════════════════════════════════════
#  Chart Builders
# ════════════════════════════════════════════════════════════════


def build_sector_heatmap(daily_records: list[dict]) -> go.Figure:
    """Bar chart of stock performance colored by sector.

    Args:
        daily_records: Daily summary dicts with symbol, sector, avg_change_pct.

    Returns:
        Plotly Figure with one bar per stock, colored by change direction.
    """
    fig = go.Figure()

    if not daily_records:
        fig.update_layout(
            title=dict(text="Sector Performance", font=dict(size=16)),
            **DARK_LAYOUT,
            annotations=[
                dict(
                    text="No data available",
                    showarrow=False,
                    font=dict(size=14, color=COLORS["muted"]),
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                )
            ],
        )
        return fig

    sorted_records = sorted(daily_records, key=lambda r: r["avg_change_pct"], reverse=True)
    symbols = [r["symbol"] for r in sorted_records]
    changes = [r["avg_change_pct"] for r in sorted_records]
    colors = [COLORS["green"] if c >= 0 else COLORS["red"] for c in changes]

    fig.add_trace(
        go.Bar(
            x=symbols,
            y=changes,
            marker_color=colors,
            text=[f"{c:+.2f}%" for c in changes],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Change: %{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text="Sector Performance — Daily Change %", font=dict(size=16)),
        yaxis_title="Change %",
        showlegend=False,
        **DARK_LAYOUT,
    )

    return fig


def build_signal_table_figure(signal_records: list[dict]) -> go.Figure:
    """Plotly Table showing buy/sell/hold signals with technicals.

    Args:
        signal_records: Market signal dicts with symbol, sector, rsi,
            macd_histogram, signal, latest_close.

    Returns:
        Plotly Figure with a Table trace.
    """
    headers = ["Symbol", "Sector", "Price", "RSI", "MACD Hist", "Signal"]

    if not signal_records:
        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=headers,
                        fill_color=COLORS["card"],
                        font=dict(color=COLORS["text"], size=13),
                        line_color=COLORS["border"],
                        align="left",
                    ),
                    cells=dict(
                        values=[[] for _ in headers],
                        fill_color=COLORS["bg"],
                        font=dict(color=COLORS["text"]),
                        line_color=COLORS["border"],
                    ),
                )
            ]
        )
        fig.update_layout(title=dict(text="Technical Signals", font=dict(size=16)), **DARK_LAYOUT)
        return fig

    sorted_records = sorted(signal_records, key=lambda r: r["symbol"])

    symbols = [r["symbol"] for r in sorted_records]
    sectors = [r["sector"] for r in sorted_records]
    prices = [f"${r['latest_close']:.2f}" for r in sorted_records]
    rsis = [f"{r['rsi']:.1f}" if r.get("rsi") is not None else "—" for r in sorted_records]
    macds = [f"{r['macd_histogram']:.4f}" if r.get("macd_histogram") is not None else "—" for r in sorted_records]
    signals = [r["signal"].upper() for r in sorted_records]

    # Color signals
    signal_colors = []
    for s in signals:
        if s == "BUY":
            signal_colors.append(COLORS["green"])
        elif s == "SELL":
            signal_colors.append(COLORS["red"])
        else:
            signal_colors.append(COLORS["yellow"])

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=headers,
                    fill_color=COLORS["card"],
                    font=dict(color=COLORS["text"], size=13),
                    line_color=COLORS["border"],
                    align="left",
                ),
                cells=dict(
                    values=[symbols, sectors, prices, rsis, macds, signals],
                    fill_color=COLORS["bg"],
                    font=dict(color=[COLORS["text"]] * 5 + [signal_colors], size=12),
                    line_color=COLORS["border"],
                    align="left",
                ),
            )
        ]
    )

    fig.update_layout(
        title=dict(text="Technical Signals — All Stocks", font=dict(size=16)),
        **DARK_LAYOUT,
    )

    return fig


def build_market_overview_chart(daily_records: list[dict]) -> go.Figure:
    """Horizontal bar chart of all stocks sorted by daily change %.

    Args:
        daily_records: Daily summary dicts with symbol, avg_change_pct.

    Returns:
        Plotly Figure with bars sorted by performance.
    """
    fig = go.Figure()

    if not daily_records:
        fig.update_layout(
            title=dict(text="Market Overview", font=dict(size=16)),
            **DARK_LAYOUT,
            annotations=[
                dict(
                    text="No data available",
                    showarrow=False,
                    font=dict(size=14, color=COLORS["muted"]),
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                )
            ],
        )
        return fig

    sorted_records = sorted(daily_records, key=lambda r: r["avg_change_pct"])
    symbols = [r["symbol"] for r in sorted_records]
    changes = [r["avg_change_pct"] for r in sorted_records]
    colors = [COLORS["green"] if c >= 0 else COLORS["red"] for c in changes]

    fig.add_trace(
        go.Bar(
            x=changes,
            y=symbols,
            orientation="h",
            marker_color=colors,
            text=[f"{c:+.2f}%" for c in changes],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Change: %{x:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text="Market Overview — Daily Performance", font=dict(size=16)),
        xaxis_title="Change %",
        showlegend=False,
        **DARK_LAYOUT,
    )

    return fig


# ════════════════════════════════════════════════════════════════
#  Dashboard Data Formatter
# ════════════════════════════════════════════════════════════════


def format_dashboard_data(daily_records: list[dict], signal_records: list[dict]) -> dict:
    """Extract display-ready metrics from raw data.

    Args:
        daily_records: Daily summary dicts.
        signal_records: Market signal dicts.

    Returns:
        Dict with top_gainers, top_losers, signal_counts, total_stocks.
    """
    if not daily_records:
        return {
            "total_stocks": 0,
            "top_gainers": [],
            "top_losers": [],
            "signal_counts": {},
        }

    sorted_by_change = sorted(daily_records, key=lambda r: r["avg_change_pct"], reverse=True)

    top_gainers = [{"symbol": r["symbol"], "change_pct": round(r["avg_change_pct"], 2)} for r in sorted_by_change[:3]]
    top_losers = [{"symbol": r["symbol"], "change_pct": round(r["avg_change_pct"], 2)} for r in sorted_by_change[-3:]]

    signal_counts: dict[str, int] = {}
    for r in signal_records:
        sig = r.get("signal", "unknown")
        signal_counts[sig] = signal_counts.get(sig, 0) + 1

    return {
        "total_stocks": len(daily_records),
        "top_gainers": top_gainers,
        "top_losers": top_losers,
        "signal_counts": signal_counts,
    }


# ════════════════════════════════════════════════════════════════
#  Chat Handler
# ════════════════════════════════════════════════════════════════


def chat_handler(message: str, history: list, agent) -> str:
    """Handle a chat message by running the MarketMind agent.

    Args:
        message: User's question.
        history: Gradio chat history (list of [user, assistant] pairs).
        agent: MarketMindAgent instance.

    Returns:
        Agent's answer as a string.
    """
    try:
        result = agent.run(message)
        return result["answer"]
    except Exception as e:
        return f"Sorry, an error occurred: {e}"
