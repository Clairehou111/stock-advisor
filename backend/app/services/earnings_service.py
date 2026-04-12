"""
Earnings calendar service — reads from price_service's unified yfinance cache.

No separate API calls needed — earnings dates are fetched as part of the
single yfinance call in get_price() / get_ticker_meta().
"""

from __future__ import annotations

import logging
from datetime import date

from app.services.price_service import get_ticker_meta

logger = logging.getLogger(__name__)


async def get_upcoming_earnings(tickers: list[str]) -> dict[str, date | None]:
    """Get upcoming earnings dates for multiple tickers.

    Reads from the meta cache populated by get_price(). If meta isn't cached yet,
    get_ticker_meta() will fetch it (single yfinance call).
    """
    results: dict[str, date | None] = {}
    for ticker in tickers:
        try:
            meta = await get_ticker_meta(ticker)
            results[ticker] = meta.earnings_date
        except Exception:
            logger.debug("No earnings data for %s", ticker)
            results[ticker] = None
    return results


def format_earnings_notice(earnings: dict[str, date | None]) -> str:
    """Format earnings dates into a notice string for the system prompt."""
    today = date.today()
    notices = []

    for ticker, earnings_date in earnings.items():
        if earnings_date is None:
            continue
        days_until = (earnings_date - today).days
        if days_until < 0:
            continue  # past earnings
        if days_until <= 7:
            notices.append(f"⚠️ **{ticker}** reports earnings in **{days_until} day{'s' if days_until != 1 else ''}** ({earnings_date.strftime('%b %d')})")
        elif days_until <= 30:
            notices.append(f"📅 {ticker} earnings: {earnings_date.strftime('%b %d')} ({days_until} days)")

    if not notices:
        return ""

    return "## Upcoming Earnings\n" + "\n".join(notices)
