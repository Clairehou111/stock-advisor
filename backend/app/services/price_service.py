"""
Price/fundamentals service — near real-time prices via Finnhub, PE via yfinance.

Cache: in-memory dict with 30-second TTL (no DB round-trip).
Zero prices are never cached — they indicate a failed fetch and should be retried.

Flow:
  1. Check in-memory cache (30s TTL, non-zero only)
  2. Cache miss: fetch price from Finnhub (real-time, 60 req/min free)
  3. Fetch PE from yfinance (15min delay is fine for fundamentals)
  4. Fallback: yfinance price if Finnhub key not configured
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings

logger = logging.getLogger(__name__)

_CACHE_TTL = 30.0  # seconds
_META_CACHE_TTL = 86400.0  # 24h — name/sector/industry rarely change


@dataclass
class PriceData:
    ticker: str
    price: float
    pe_ratio: float | None


@dataclass
class TickerMeta:
    """Cached yfinance metadata for semantic retrieval bridge."""
    long_name: str
    sector: str
    industry: str


@dataclass
class _CacheEntry:
    data: PriceData
    fetched_at: float = field(default_factory=time.monotonic)

    def is_fresh(self) -> bool:
        return time.monotonic() - self.fetched_at < _CACHE_TTL


@dataclass
class _MetaCacheEntry:
    data: TickerMeta
    fetched_at: float = field(default_factory=time.monotonic)

    def is_fresh(self) -> bool:
        return time.monotonic() - self.fetched_at < _META_CACHE_TTL


# Module-level in-memory caches
_cache: dict[str, _CacheEntry] = {}
_meta_cache: dict[str, _MetaCacheEntry] = {}


async def _fetch_finnhub(ticker: str) -> float:
    """Fetch real-time price from Finnhub. Returns 0.0 on failure."""
    url = "https://finnhub.io/api/v1/quote"
    async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
        resp = await client.get(url, params={"symbol": ticker, "token": settings.finnhub_api_key})
        resp.raise_for_status()
        data = resp.json()
    price = data.get("c") or data.get("pc") or 0.0
    return float(price)


def _fetch_pe_yfinance(ticker: str) -> float | None:
    """Synchronous yfinance fetch for PE ratio only."""
    import yfinance as yf
    info = yf.Ticker(ticker).info or {}
    pe = info.get("trailingPE") or info.get("forwardPE")
    return float(pe) if pe else None


def _fetch_yfinance_full(ticker: str) -> PriceData:
    """Synchronous full yfinance fetch (price + PE) — used when Finnhub unavailable."""
    import yfinance as yf
    info = yf.Ticker(ticker).info or {}
    price = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0
    pe = info.get("trailingPE") or info.get("forwardPE")
    return PriceData(ticker=ticker, price=float(price), pe_ratio=float(pe) if pe else None)


async def get_price(ticker: str, db: AsyncSession | None = None) -> PriceData:
    """
    Get price + PE for a ticker.

    Uses a 30-second in-memory cache. Zero prices are never cached.
    The `db` parameter is accepted for backwards compatibility but unused.
    """
    entry = _cache.get(ticker)
    if entry and entry.is_fresh():
        return entry.data

    try:
        if settings.finnhub_api_key:
            price_task = _fetch_finnhub(ticker)
            pe_task = asyncio.to_thread(_fetch_pe_yfinance, ticker)
            price, pe_ratio = await asyncio.gather(price_task, pe_task, return_exceptions=True)

            if isinstance(price, Exception):
                logger.warning("Finnhub failed for %s: %s — falling back to yfinance", ticker, price)
                data = await asyncio.to_thread(_fetch_yfinance_full, ticker)
            else:
                pe_ratio = None if isinstance(pe_ratio, Exception) else pe_ratio
                data = PriceData(ticker=ticker, price=price, pe_ratio=pe_ratio)
        else:
            data = await asyncio.to_thread(_fetch_yfinance_full, ticker)
    except Exception:
        logger.exception("Price fetch failed for %s", ticker)
        # Return stale cache if available, else zero
        if entry:
            return entry.data
        return PriceData(ticker=ticker, price=0.0, pe_ratio=None)

    # Only cache successful fetches — zero means fetch failed
    if data.price > 0:
        _cache[ticker] = _CacheEntry(data=data)

    return data


def _fetch_yfinance_meta(ticker: str) -> TickerMeta:
    """Synchronous yfinance fetch for name/sector/industry metadata."""
    import yfinance as yf
    info = yf.Ticker(ticker).info or {}
    return TickerMeta(
        long_name=info.get("longName") or info.get("shortName") or ticker,
        sector=info.get("sector") or "",
        industry=info.get("industry") or "",
    )


async def get_ticker_meta(ticker: str) -> TickerMeta:
    """Get ticker metadata (name, sector, industry). Cached for 24h."""
    entry = _meta_cache.get(ticker)
    if entry and entry.is_fresh():
        return entry.data
    try:
        meta = await asyncio.to_thread(_fetch_yfinance_meta, ticker)
        _meta_cache[ticker] = _MetaCacheEntry(data=meta)
        return meta
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
