"""Shared retry helper for all external LLM / API calls."""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

logger = logging.getLogger(__name__)

_RETRYABLE = ("503", "429", "unavailable", "overloaded", "timeout", "rate_limit", "rate limit", "capacity", "high demand", "resource exhausted", "quota")
MAX_RETRIES = 3
BASE_DELAY = 2.0


def _is_retryable(err: Exception) -> bool:
    err_lower = str(err).lower()
    return any(s in err_lower for s in _RETRYABLE)


async def retry(
    fn: Callable,
    *args,
    max_retries: int = MAX_RETRIES,
    label: str = "LLM call",
    sync: bool = False,
    **kwargs,
):
    """Retry a function on transient errors with exponential backoff.

    Works for both async and sync functions:
      - async fn: awaited directly
      - sync fn (sync=True): run via asyncio.to_thread, retries use asyncio.sleep
    """
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            if sync:
                return await asyncio.to_thread(fn, *args, **kwargs)
            return await fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            if _is_retryable(e) and attempt < max_retries - 1:
                delay = BASE_DELAY * (2 ** attempt)
                logger.warning("%s attempt %d failed (%s), retrying in %.0fs…", label, attempt + 1, str(e)[:120], delay)
                await asyncio.sleep(delay)
                continue
            raise
    raise last_err  # type: ignore[misc]
