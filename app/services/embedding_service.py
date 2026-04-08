"""
Embedding service — Gemini text-embedding-004 (768-dim).

Uses the google-genai SDK which handles auth, retries, and proxy correctly.
"""

from __future__ import annotations

import asyncio
import logging

from app.config import settings

logger = logging.getLogger(__name__)

MODEL = "models/gemini-embedding-001"
DIMENSIONS = 768
BATCH_LIMIT = 100


def _embed_batch_sync(texts: list[str]) -> list[list[float]]:
    """Synchronous batch embedding via google-genai SDK."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key.strip())

    result = client.models.embed_content(
        model=MODEL,
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=DIMENSIONS),
    )

    return [list(e.values) for e in result.embeddings]


async def embed_text(text: str) -> list[float]:
    """Embed a single text string. Returns 768-dim vector."""
    results = await embed_batch([text])
    return results[0]


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed multiple texts. Handles batching if > BATCH_LIMIT.

    Returns list of 768-dim vectors in same order as input.
    """
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY not configured — cannot generate embeddings")

    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), BATCH_LIMIT):
        batch = texts[i : i + BATCH_LIMIT]
        embeddings = await asyncio.to_thread(_embed_batch_sync, batch)
        all_embeddings.extend(embeddings)

    return all_embeddings
