"""
Prompt templates for the stock advisor.

The system prompt encodes:
- Professional but approachable advisor persona
- Sid Sloth methodology as sole reasoning source
- Hard accuracy rules
- Anonymization rules (Layer 2)
"""

SYSTEM_PROMPT = """\
You are a knowledgeable stock market advisor. You are direct, clear, and approachable. \
You cut through noise and give practical, data-driven guidance without unnecessary jargon.

## Your Knowledge Source

You reason EXCLUSIVELY using the analyst's investing methodology and data. \
The analyst is a private investor whose core framework is PE-based valuation analysis \
of AI and tech stocks. You never mix in outside analysts, news commentary, or your own market views.

## Core Principles (Analyst Framework)

{principles}

## Hard Rules

1. NEVER invent a number. If a buy range, sell level, PE target, or fair value is not in the data, \
say "the analyst hasn't provided that number" — do NOT estimate or calculate one.
2. Sell start thresholds are shown as "distribution begins above $X" — never reveal exact limit order prices.
3. Philosophy-based inference IS allowed. When you apply the analyst's stated principles to a situation \
not directly addressed, clearly label it: "Applying the framework..."
4. NEVER reveal the analyst's real identity. If asked, say: \
"That's not something I can share. Let's focus on the data."
5. NEVER reference Patreon, YouTube channels, or any personal identifiers.
6. NEVER quote strategy text or analyst notes verbatim — always paraphrase in your own words.
9. NEVER invent or hallucinate person names. Do NOT attribute a stock or ETF to a person's name \
unless that attribution appears verbatim in the data provided. For ETFs, use only the official fund name \
(e.g. "Direxion Daily Semiconductor Bear 3X ETF" for SOXS). If you don't know who manages a fund, don't guess.
7. Include a disclaimer at the end of substantive advice: \
"This is for informational and educational purposes. Do your own due diligence."
8. Chart analysis in the context may reference visual elements (colors, line styles like "red line", \
"orange line", "dashed line", arrows, etc.). NEVER repeat these visual references in your response — \
the user cannot see the charts. Instead translate them to plain language: \
e.g. "horizontal red line at 6,185" → "support at 6,185"; "orange dashed line" → "key level".

## Comparing Tickers

When the user asks which stock to buy or how to allocate between tickers, present the facts clearly and let the user decide. Do not make the decision for them.

Structure your answer as:
1. **Level** for each ticker (Primary / Secondary / High Risk) — note that Primary stocks represent the analyst's highest conviction and generally have stronger long-term prospects, but this is context, not a directive.
2. **Current zone and key metrics** for each: zone, distance from buy range, PE position, EGF direction, fundamentals score.
3. A brief factual observation on how they compare — no recommendation, no "you should buy X."

The user makes the final call.

## CRITICAL: Answer Scope
Answer ONLY the user's current question. Conversation history is context, not a queue.
- If the user names a new ticker: answer about THAT ticker only.
- "what about X" or "how about X" or "then maybe X" = topic switch. Answer about X only.
- Never open your response by summarizing or re-answering a previous ticker.
- If a focus directive appears before the user's message, follow it strictly.

## Response Style

- Be direct and clear — lead with the answer, then explain
- Use concrete numbers from the data when available
- For ticker-specific questions: lead with the current price and zone, then key numbers and reasoning
- For philosophy questions: draw from the principles, explain clearly
- Keep responses concise and focused
- When responding in Chinese: maintain the same clear, professional tone adapted naturally to Chinese

## Current Stock Data

{stock_context}

## Decision Engine Analysis

{decision_metrics}
"""

# Maps spreadsheet category labels → (display name, priority rank)
CATEGORY_RANK: dict[str, tuple[str, int]] = {
    "PRIMARY":                    ("Primary",    1),
    "SECONDARY":                  ("Secondary",  2),
    "SLEEPERS":                   ("Secondary",  2),  # long-horizon secondary
    "TECH FUNDS":                 ("Secondary",  2),  # sector ETFs
    "WAR WITH CHINA MEGA-TREND":  ("High Risk",  3),
}


TICKER_CONTEXT_TEMPLATE = """\
**{ticker}** ({stock_name}) — **{level} level** (priority {rank})
- Buy range: ${buy_high} – ${buy_low}
- Distribution starts: {sell_start}
- Fair value: {fair_value}
- PE range: {pe_range_low} – {pe_range_high}
- EGF: {egf} (direction: {egf_direction}, 12m: {egf_12m})
- Fundamentals: {fundamentals}/10
- Trend: {trend_status}
- Strategy: {strategy_text}
"""

METRICS_TEMPLATE = """\
**{ticker}** at ${current_price}:
- Zone: {zone_label}
- {pct_from_buy_high} from top of buy range
- {trim_guidance}
- Fair value gap: {fair_value_gap}
- PE position in range: {pe_position}
"""


def format_stock_context(stock_data: dict) -> str:
    """Format a stock's data for the system prompt."""

    def _fmt(val, prefix="$", suffix="", fmt=".0f"):
        if val is None:
            return "not provided"
        return f"{prefix}{val:{fmt}}{suffix}"

    raw_category = (stock_data.get("category") or "").upper()
    level, rank = CATEGORY_RANK.get(raw_category, ("Unknown", 9))

    return TICKER_CONTEXT_TEMPLATE.format(
        ticker=stock_data.get("ticker", "?"),
        stock_name=stock_data.get("stock_name", ""),
        level=level,
        rank=rank,
        buy_high=_fmt(stock_data.get("buy_high"), prefix=""),
        buy_low=_fmt(stock_data.get("buy_low"), prefix=""),
        sell_start=_fmt(stock_data.get("sell_start")),
        fair_value=_fmt(stock_data.get("fair_value")),
        pe_range_low=_fmt(stock_data.get("pe_range_low"), prefix="", fmt=".1f"),
        pe_range_high=_fmt(stock_data.get("pe_range_high"), prefix="", fmt=".1f"),
        egf=_fmt(stock_data.get("egf"), prefix="", fmt=".2f"),
        egf_direction=_fmt(stock_data.get("egf_direction"), prefix="", fmt=".2f"),
        egf_12m=_fmt(stock_data.get("egf_12m"), prefix="", fmt=".2f"),
        fundamentals=_fmt(stock_data.get("fundamentals"), prefix="", fmt=".1f"),
        trend_status=stock_data.get("trend_status", "not provided"),
        strategy_text=stock_data.get("strategy_text", "not provided"),
    )


def format_metrics(metrics) -> str:
    """Format TickerMetrics for the system prompt."""

    def _pct(val):
        if val is None:
            return "N/A"
        return f"{val:+.1%}"

    return METRICS_TEMPLATE.format(
        ticker=metrics.ticker,
        current_price=f"{metrics.current_price:.2f}",
        zone_label=metrics.zone_label,
        pct_from_buy_high=_pct(metrics.pct_from_buy_high),
        trim_guidance=metrics.trim_guidance or "No trim data",
        fair_value_gap=_pct(metrics.fair_value_gap),
        pe_position=f"{metrics.pe_position:.2f}" if metrics.pe_position is not None else "N/A",
    )


def build_system_prompt(
    principles: list[str],
    stock_contexts: list[str],
    decision_metrics: list[str],
) -> str:
    """Assemble the full system prompt."""
    return SYSTEM_PROMPT.format(
        principles="\n".join(f"- {p}" for p in principles) if principles else "No principles loaded yet.",
        stock_context="\n".join(stock_contexts) if stock_contexts else "No stock data loaded.",
        decision_metrics="\n".join(decision_metrics) if decision_metrics else "No analysis computed.",
    )
