"""
Embedding service — 768-dim vectors via Gemini gemini-embedding-001.

Primary:  Direct Gemini API
Fallback: Same model via OpenRouter (different infrastructure route)

Both retry on transient errors. If both routes fail, returns None
so callers can store chunks without vectors and backfill later.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from app.config import settings
from app.llm.retry import retry

logger = logging.getLogger(__name__)

MODEL = "models/gemini-embedding-2-preview"
OPENROUTER_MODEL = "qwen/qwen3-embedding-4b"
DIMENSIONS = 1024
BATCH_LIMIT = 100


def _embed_via_gemini(texts: list[str]) -> list[list[float]]:
    """Embed via direct Gemini API."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key.strip())
    result = client.models.embed_content(
        model=MODEL,
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=DIMENSIONS),
    )
    return [list(e.values) for e in result.embeddings]


def _embed_via_openrouter(texts: list[str]) -> list[list[float]]:
    """Embed via OpenRouter (OpenAI-compatible embeddings endpoint)."""
    api_key = settings.openrouter_api_key
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not configured")

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": OPENROUTER_MODEL,
                "input": texts,
                "dimensions": DIMENSIONS,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    sorted_items = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in sorted_items]


async def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Batch embedding: direct Gemini (retry) → OpenRouter (retry)."""
    if settings.gemini_api_key:
        try:
            return await retry(
                _embed_via_gemini, texts,
                max_retries=3, label="Embedding/Gemini-direct", sync=True,
            )
        except Exception as e:
            logger.warning("Direct Gemini embedding failed after retries (%s), trying OpenRouter", e)

    if settings.openrouter_api_key:
        try:
            return await retry(
                _embed_via_openrouter, texts,
                max_retries=3, label="Embedding/OpenRouter", sync=True,
            )
        except Exception as e:
            logger.error("OpenRouter embedding also failed after retries: %s", e)
            raise

    raise RuntimeError("No embedding provider available — configure GEMINI_API_KEY or OPENROUTER_API_KEY")


async def embed_text(text: str) -> list[float] | None:
    """Embed a single text string. Returns 768-dim vector or None on failure."""
    results = await embed_batch([text])
    return results[0]


async def embed_batch(texts: list[str]) -> list[list[float] | None]:
    """Embed multiple texts. Handles batching if > BATCH_LIMIT.

    Returns None for texts whose embeddings failed after both routes exhausted.
    """
    if not settings.gemini_api_key and not settings.openrouter_api_key:
        logger.warning("No embedding API key configured — returning None embeddings")
        return [None] * len(texts)

    all_embeddings: list[list[float] | None] = []

    for i in range(0, len(texts), BATCH_LIMIT):
        batch = texts[i : i + BATCH_LIMIT]
        try:
            embeddings = await _embed_batch(batch)
            all_embeddings.extend(embeddings)
        except Exception as e:
            logger.error(
                "All embedding routes failed (%s) — storing %d chunks without vectors",
                e, len(batch),
            )
            all_embeddings.extend([None] * len(batch))

    return all_embeddings
