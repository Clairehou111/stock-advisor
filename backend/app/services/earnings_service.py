"""
Earnings calendar service — fetches upcoming earnings dates via yfinance.

Results are cached daily (earnings dates don't change frequently).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# In-memory cache: {ticker: (earnings_date, fetched_at)}
_cache: dict[str, tuple[date | None, datetime]] = {}
CACHE_TTL = timedelta(hours=24)


def _fetch_earnings_date(ticker: str) -> date | None:
    """Synchronous yfinance fetch — run via asyncio.to_thread()."""
    import yfinance as yf

    try:
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        if cal is not None and not cal.empty:
            # calendar returns a DataFrame with 'Earnings Date' row
            earnings_dates = cal.loc["Earnings Date"] if "Earnings Date" in cal.index else None
            if earnings_dates is not None:
                # First column is the next earnings date
                next_date = earnings_dates.iloc[0]
                if hasattr(next_date, "date"):
                    return next_date.date()
                return None
    except Exception:
        logger.debug("No earnings data for %s", ticker)

    return None


async def get_upcoming_earnings(tickers: list[str]) -> dict[str, date | None]:
    """
    Get upcoming earnings dates for multiple tickers.

    Returns dict mapping ticker → next earnings date (or None).
    """
    now = datetime.now(timezone.utc)
    results: dict[str, date | None] = {}

    for ticker in tickers:
        # Check cache
        if ticker in _cache:
            cached_date, fetched_at = _cache[ticker]
            if now - fetched_at < CACHE_TTL:
                results[ticker] = cached_date
                continue

        # Fetch fresh
        try:
            earnings_date = await asyncio.to_thread(_fetch_earnings_date, ticker)
            _cache[ticker] = (earnings_date, now)
            results[ticker] = earnings_date
        except Exception:
            logger.exception("Failed to fetch earnings for %s", ticker)
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
            continue
        if days_until == 0:
            notices.append(f"**{ticker}** reports earnings TODAY")
        elif days_until <= 7:
            notices.append(f"**{ticker}** reports earnings in {days_until} day{'s' if days_until != 1 else ''} ({earnings_date.isoformat()})")
        elif days_until <= 30:
            notices.append(f"{ticker} earnings: {earnings_date.isoformat()}")

    if not notices:
        return ""

    return "## Upcoming Earnings\n" + "\n".join(f"- {n}" for n in notices)
