"""
Decision Engine — Stage 1: Deterministic Math (no LLM).

Computes zones, trimming guidance, PE position, and FOMO indicator
from stock_predictions data + live price. All math uses only
explicitly stated analyst data — missing fields stay null.

Stage 2 (LLM reasoning) is handled in app/llm/.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Zone(str, Enum):
    DEEP_VALUE = "deep_value"
    ACCUMULATION = "accumulation"
    HOLD = "hold"
    APPROACHING_DISTRIBUTION = "approaching_distribution"
    DISTRIBUTION = "distribution"
    UNKNOWN = "unknown"


@dataclass
class TickerMetrics:
    """Computed metrics for one ticker at a given price."""

    ticker: str
    current_price: float

    # Zone
    zone: Zone
    zone_label: str  # analyst's own label if available, else zone.value

    # Buy range metrics
    pct_from_buy_high: float | None = None  # positive = above buy range, negative = below
    pct_from_buy_low: float | None = None
    buy_range_position: float | None = None  # 0.0 = at buy_low, 1.0 = at buy_high, >1 = above

    # Sell / trim
    pct_from_sell_start: float | None = None  # positive = above sell_start
    trim_guidance: str | None = None

    # Fair value
    fair_value_gap: float | None = None  # (price - fair_value) / fair_value

    # PE metrics
    pe_position: float | None = None  # 0.0 = at pe_low, 1.0 = at pe_high
    fomo_indicator: float | None = None

    # EGF
    egf: float | None = None
    egf_direction: float | None = None
    egf_12m: float | None = None

    # Fundamentals
    fundamentals: float | None = None
    trend_status: str | None = None
    prob_new_ath: float | None = None


@dataclass
class StockData:
    """Input data for one ticker — from stock_predictions table."""

    ticker: str
    stock_name: str | None = None
    buy_high: float | None = None
    buy_low: float | None = None
    sell_start: float | None = None
    pe_range_high: float | None = None
    pe_range_low: float | None = None
    fair_value: float | None = None
    egf: float | None = None
    egf_direction: float | None = None
    egf_12m: float | None = None
    fundamentals: float | None = None
    trend_status: str | None = None
    prob_new_ath: float | None = None
    strategy_text: str | None = None
    analyst_labels: dict | None = None


def _pct_diff(price: float, reference: float) -> float:
    """Percentage difference: (price - reference) / reference."""
    if reference == 0:
        return 0.0
    return (price - reference) / reference


def detect_zone(price: float, data: StockData) -> tuple[Zone, str]:
    """
    Determine which zone the current price falls in.

    Uses analyst-defined labels when available.
    Returns (Zone enum, human-readable label).
    """
    labels = data.analyst_labels or {}

    # Need at least buy_high to determine zones
    if data.buy_high is None:
        return Zone.UNKNOWN, "insufficient data"

    # Below buy_low (if exists) = deep value
    if data.buy_low is not None and price < data.buy_low:
        label = labels.get("buy_low", "deep value")
        return Zone.DEEP_VALUE, label

    # Within buy range
    if data.buy_low is not None and price <= data.buy_high:
        return Zone.ACCUMULATION, "accumulation zone"
    if data.buy_low is None and price <= data.buy_high:
        return Zone.ACCUMULATION, "accumulation zone"

    # Above buy range
    if data.sell_start is not None and price >= data.sell_start:
        return Zone.DISTRIBUTION, "distribution zone"

    if data.fair_value is not None and data.sell_start is not None:
        if price >= data.fair_value:
            return Zone.APPROACHING_DISTRIBUTION, "hold, approaching distribution"

    return Zone.HOLD, "hold"


def compute_trim_guidance(price: float, data: StockData) -> str | None:
    """
    Generate trimming guidance based on analyst's stated rules.

    Rule: ~5% of position for every 10% advance above sell_start.
    sell_start is surfaced as a threshold — exact limit order prices withheld.
    """
    if data.sell_start is None:
        return None

    if price < data.sell_start:
        pct_to = _pct_diff(data.sell_start, price)
        return f"Distribution begins above ${data.sell_start:.0f} ({pct_to:.0%} away)"

    # Price is above sell_start — compute trim percentage
    pct_above = _pct_diff(price, data.sell_start)
    # ~5% trim per 10% advance
    trim_pct = min(100, int((pct_above / 0.10) * 5))
    return (
        f"In distribution zone — price is {pct_above:.1%} above ${data.sell_start:.0f}. "
        f"Consider trimming ~{trim_pct}% of position based on stated trimming rules."
    )


def analyze_ticker(price: float, data: StockData, current_pe: float | None = None) -> TickerMetrics:
    """
    Full Stage 1 analysis for one ticker at a given price.

    Args:
        price: Current market price
        data: Analyst's stated data for this ticker
        current_pe: Live P/E ratio (from price feed)
    """
    zone, zone_label = detect_zone(price, data)

    metrics = TickerMetrics(
        ticker=data.ticker,
        current_price=price,
        zone=zone,
        zone_label=zone_label,
        egf=data.egf,
        egf_direction=data.egf_direction,
        egf_12m=data.egf_12m,
        fundamentals=data.fundamentals,
        trend_status=data.trend_status,
        prob_new_ath=data.prob_new_ath,
    )

    # Buy range metrics
    if data.buy_high is not None:
        metrics.pct_from_buy_high = _pct_diff(price, data.buy_high)
    if data.buy_low is not None:
        metrics.pct_from_buy_low = _pct_diff(price, data.buy_low)
    if data.buy_high is not None and data.buy_low is not None and data.buy_high != data.buy_low:
        metrics.buy_range_position = (price - data.buy_low) / (data.buy_high - data.buy_low)

    # Sell / trim
    if data.sell_start is not None:
        metrics.pct_from_sell_start = _pct_diff(price, data.sell_start)
    metrics.trim_guidance = compute_trim_guidance(price, data)

    # Fair value
    if data.fair_value is not None:
        metrics.fair_value_gap = _pct_diff(price, data.fair_value)

    # PE position within analyst's stated range
    if (
        current_pe is not None
        and data.pe_range_high is not None
        and data.pe_range_low is not None
        and data.pe_range_high != data.pe_range_low
    ):
        metrics.pe_position = (current_pe - data.pe_range_low) / (
            data.pe_range_high - data.pe_range_low
        )

    # FOMO indicator: conceptually current_pe% - forward_pe%
    # We use pe_position as proxy when we have it
    if metrics.pe_position is not None:
        metrics.fomo_indicator = metrics.pe_position  # >1.0 = above historical PE high = FOMO

    return metrics
