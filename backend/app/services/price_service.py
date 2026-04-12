"""
Price/fundamentals service — near real-time prices via Finnhub, everything else via yfinance.

Single yfinance call per ticker returns: price (fallback), PE, earnings date, name/sector/industry.
Finnhub provides real-time price; yfinance provides fundamentals + metadata.

Cache: in-memory with 30s TTL for price data, 24h for metadata.
Zero prices are never cached — they indicate a failed fetch and should be retried.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import date

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings

logger = logging.getLogger(__name__)

_PRICE_CACHE_TTL = 30.0  # seconds
_META_CACHE_TTL = 86400.0  # 24h — name/sector/industry/earnings rarely change


@dataclass
class PriceData:
    ticker: str
    price: float
    pe_ratio: float | None


@dataclass
class TickerMeta:
    """Cached yfinance metadata for semantic retrieval bridge + earnings."""
    long_name: str
    sector: str
    industry: str
    earnings_date: date | None = None


@dataclass
class _PriceCacheEntry:
    data: PriceData
    fetched_at: float = field(default_factory=time.monotonic)

    def is_fresh(self) -> bool:
        return time.monotonic() - self.fetched_at < _PRICE_CACHE_TTL


@dataclass
class _MetaCacheEntry:
    data: TickerMeta
    fetched_at: float = field(default_factory=time.monotonic)

    def is_fresh(self) -> bool:
        return time.monotonic() - self.fetched_at < _META_CACHE_TTL


# Module-level in-memory caches
_price_cache: dict[str, _PriceCacheEntry] = {}
_meta_cache: dict[str, _MetaCacheEntry] = {}


# ── Finnhub (real-time price only) ───────────────────────────────────────────


async def _fetch_finnhub(ticker: str) -> float:
    """Fetch real-time price from Finnhub. Returns 0.0 on failure."""
    url = "https://finnhub.io/api/v1/quote"
    async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
        resp = await client.get(url, params={"symbol": ticker, "token": settings.finnhub_api_key})
        resp.raise_for_status()
        data = resp.json()
    price = data.get("c") or data.get("pc") or 0.0
    return float(price)


# ── yfinance (single call: price fallback + PE + earnings + metadata) ────────


def _fetch_yfinance_all(ticker: str) -> dict:
    """Single yfinance call returning everything we need.

    Returns dict with: price, pe, long_name, sector, industry, earnings_date
    """
    import yfinance as yf
    tk = yf.Ticker(ticker)
    info = tk.info or {}

    # Price + PE
    price = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0
    pe = info.get("trailingPE") or info.get("forwardPE")

    # Metadata
    long_name = info.get("longName") or info.get("shortName") or ticker
    sector = info.get("sector") or ""
    industry = info.get("industry") or ""

    # Earnings date
    earnings_date = None
    try:
        cal = tk.calendar
        if cal is not None and not cal.empty and "Earnings Date" in cal.index:
            next_date = cal.loc["Earnings Date"].iloc[0]
            if hasattr(next_date, "date"):
                earnings_date = next_date.date()
    except Exception:
        pass

    return {
        "price": float(price),
        "pe": float(pe) if pe else None,
        "long_name": long_name,
        "sector": sector,
        "industry": industry,
        "earnings_date": earnings_date,
    }


# ── Public API ───────────────────────────────────────────────────────────────


async def get_price(ticker: str, db: AsyncSession | None = None) -> PriceData:
    """Get price + PE for a ticker. Uses 30s cache. Finnhub for real-time price,
    yfinance for PE (fetched together via get_ticker_data)."""
    entry = _price_cache.get(ticker)
    if entry and entry.is_fresh():
        return entry.data

    try:
        if settings.finnhub_api_key:
            # Finnhub for real-time price, yfinance for PE (parallel)
            price_task = _fetch_finnhub(ticker)
            yf_task = asyncio.to_thread(_fetch_yfinance_all, ticker)
            price, yf_data = await asyncio.gather(price_task, yf_task, return_exceptions=True)

            if isinstance(price, Exception):
                logger.warning("Finnhub failed for %s: %s — using yfinance price", ticker, price)
                if isinstance(yf_data, Exception):
                    raise yf_data
                price = yf_data["price"]

            if isinstance(yf_data, Exception):
                logger.warning("yfinance failed for %s: %s", ticker, yf_data)
                pe_ratio = None
            else:
                pe_ratio = yf_data["pe"]
                # Cache metadata from the same yfinance call (free, already fetched)
                _cache_meta_from_yf(ticker, yf_data)

            data = PriceData(ticker=ticker, price=price, pe_ratio=pe_ratio)
        else:
            yf_data = await asyncio.to_thread(_fetch_yfinance_all, ticker)
            _cache_meta_from_yf(ticker, yf_data)
            data = PriceData(ticker=ticker, price=yf_data["price"], pe_ratio=yf_data["pe"])

    except Exception:
        logger.exception("Price fetch failed for %s", ticker)
        if entry:
            return entry.data
        return PriceData(ticker=ticker, price=0.0, pe_ratio=None)

    if data.price > 0:
        _price_cache[ticker] = _PriceCacheEntry(data=data)

    return data


def _cache_meta_from_yf(ticker: str, yf_data: dict) -> None:
    """Cache metadata + earnings from a yfinance_all result."""
    meta = TickerMeta(
        long_name=yf_data["long_name"],
        sector=yf_data["sector"],
        industry=yf_data["industry"],
        earnings_date=yf_data.get("earnings_date"),
    )
    _meta_cache[ticker] = _MetaCacheEntry(data=meta)


async def get_ticker_meta(ticker: str) -> TickerMeta:
    """Get ticker metadata (name, sector, industry, earnings). Cached for 24h.

    If get_price() was called first, meta is already cached from the same yfinance call.
    """
    entry = _meta_cache.get(ticker)
    if entry and entry.is_fresh():
        return entry.data
    try:
        yf_data = await asyncio.to_thread(_fetch_yfinance_all, ticker)
        _cache_meta_from_yf(ticker, yf_data)
        return _meta_cache[ticker].data
    except Exception:
        logger.warning("yfinance meta fetch failed for %s", ticker)
        if entry:
            return entry.data
        return TickerMeta(long_name=ticker, sector="", industry="")


async def get_prices_batch(tickers: list[str], db: AsyncSession | None = None) -> dict[str, PriceData]:
    """Fetch prices for multiple tickers concurrently."""
    tasks = [get_price(t) for t in tickers]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    prices = {}
    for ticker, result in zip(tickers, results):
        if isinstance(result, Exception):
            logger.error("Failed to fetch price for %s: %s", ticker, result)
            prices[ticker] = PriceData(ticker=ticker, price=0.0, pe_ratio=None)
        else:
            prices[ticker] = result

    return prices
