"""
Document ingestion — parse PDF/text files into analyst_chunks via single Gemini call.

Flow:
    1. Read file (PDF via pypdf or plain text)
    2. Anonymize raw text
    3. Single Gemini 3.1 Pro call: full text → structured JSON chunks
    4. Embed all chunks
    5. Store in analyst_chunks + upload_sources
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.ingestion.anonymizer import Anonymizer
from app.models.tables import AnalystChunk, UploadSource
from app.services.embedding_service import embed_batch

logger = logging.getLogger(__name__)


_DOC_INGESTION_PROMPT = """\
Analyze this investment document. Return a JSON object with ALL information extracted.

DOCUMENT:
{text}

Return a JSON object:
{{
  "chunks": [
    {{
      "section": "topic or section title",
      "content": "Rephrased in neutral third-person analyst language. Keep ALL price levels, \
percentages, timeframes, ticker symbols, and directional signals. Remove first-person voice, \
slang, personal opinion. Do NOT over-summarize — preserve the detail and reasoning.",
      "primary_ticker": "NVDA or null if multi-topic",
      "tickers_mentioned": ["NVDA", "MSFT"],
      "chunk_type": "prediction|philosophy|commentary|egf_explanation",
      "temporal_scope": "short_term|long_term|general",
      "thesis_direction": "bullish|bearish|neutral|mixed",
      "key_levels": [
        {{"price": 150, "type": "support|resistance|buy|sell|target", \
"significance": "minor|major|critical", "note": "optional"}}
      ]
    }}
  ]
}}

RULES:
- Use real tradeable ticker symbols (^GSPC for S&P 500, BTC-USD for Bitcoin, etc.)
- When a section has specific price levels for multiple stocks, create separate chunks per stock
- Keep historical data tables and drawdown data as their own chunks
- Remove author names, personal URLs, identifying information
- Preserve full analytical detail
- Return ONLY valid JSON, no markdown fences"""


def read_file(file_path: str | Path) -> str:
    """Read a document file (PDF or text)."""
    path = Path(file_path)

    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(pages)
        except ImportError:
            raise RuntimeError("pypdf is required for PDF ingestion: pip install pypdf")

    return path.read_text(encoding="utf-8")


async def _call_gemini_ingest(text: str, progress_cb=None) -> dict:
    """Single Gemini 3.1 Pro call: full document → structured JSON."""
    from google import genai
    from google.genai import types
    from app.llm.retry import retry

    async def _log(msg: str) -> None:
        logger.info(msg)
        if progress_cb:
            await progress_cb(msg)

    prompt = _DOC_INGESTION_PROMPT.format(text=text[:100_000])  # cap at ~25K tokens

    await _log("Processing document with Gemini 3.1 Pro...")

    def _call():
        client = genai.Client(api_key=settings.gemini_api_key.strip())
        response = client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=65536,
                temperature=0.1,
            ),
        )
        return response.text.strip()

    raw = await retry(_call, max_retries=3, label="Gemini/doc-ingest", sync=True)

    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    result = json.loads(raw)
    chunk_count = len(result.get("chunks", []))
    await _log(f"Gemini returned {chunk_count} chunks")
    return result


async def ingest_document(
    file_path: str | Path,
    db: AsyncSession,
    publish_date: date | None = None,
    progress_cb=None,
    r2_key: str | None = None,
) -> list[AnalystChunk]:
    """Full document ingestion pipeline."""
    async def _progress(msg: str) -> None:
        logger.info(msg)
        if progress_cb:
            await progress_cb(msg)

    path = Path(file_path)
    await _progress(f"Reading file: {path.name}")

    # 1. Read file
    raw_text = read_file(path)
    if not raw_text.strip():
        logger.warning("Empty document: %s", path.name)
        return []

    # 2. Anonymize
    anonymizer = Anonymizer()
    raw_text = anonymizer.scrub(raw_text).text

    await _progress(f"Document: {len(raw_text):,} chars")

    # 3. Single Gemini call
    result = await _call_gemini_ingest(raw_text, progress_cb=progress_cb)

    # 4. Create upload source
    upload = UploadSource(
        file_type=path.suffix.lstrip("."),
        r2_key=r2_key,
        extracted_json={
            "chunk_count": len(result.get("chunks", [])),
            "file_name": path.name,
        },
    )
    db.add(upload)
    await db.flush()

    # 5. Parse chunks
    chunks_data = result.get("chunks", [])
    created_chunks: list[AnalystChunk] = []

    for c in chunks_data:
        chunk = AnalystChunk(
            upload_source_id=upload.id,
            ticker=c.get("primary_ticker"),
            chunk_type=c.get("chunk_type", "commentary"),
            content_text=c.get("content", ""),
            temporal_scope=c.get("temporal_scope", "general"),
            publish_date=publish_date,
            tickers_mentioned=c.get("tickers_mentioned") or None,
            thesis_direction=c.get("thesis_direction", "neutral"),
            outlook_horizon=None,
            metadata_json={
                "section": c.get("section", ""),
                "key_levels": c.get("key_levels", []),
            },
        )
        db.add(chunk)
        created_chunks.append(chunk)

    # 6. Embed all chunks
    if created_chunks:
        await _progress(f"Embedding {len(created_chunks)} chunks...")
        texts = [c.content_text for c in created_chunks]
        embeddings = await embed_batch(texts)
        for chunk, emb in zip(created_chunks, embeddings):
            chunk.embedding = emb

    await db.flush()
    await _progress(f"Saved {len(created_chunks)} chunks to database.")

    return created_chunks
