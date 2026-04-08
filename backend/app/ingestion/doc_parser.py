"""
Document ingestion — parse PDF/text files into analyst_chunks with metadata.

Flow:
    1. Read file (PDF via pypdf or plain text)
    2. Split into 300–500 token chunks at paragraph boundaries
    3. Extract metadata per chunk via Gemini Flash
    4. Anonymize via Anonymizer.scrub()
    5. Embed via embedding service
    6. Store in analyst_chunks with all metadata
    7. Create upload_sources record
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import date
from pathlib import Path

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.ingestion.anonymizer import Anonymizer
from app.models.tables import AnalystChunk, UploadSource
from app.services.embedding_service import embed_batch

logger = logging.getLogger(__name__)

_REPHRASE_PROMPT = """\
Rephrase the following investment analysis text in neutral, third-person language \
as a generic quantitative analyst would write it. \
Keep all tickers, price levels, percentage moves, timeframes, and directional signals intact. \
Remove any first-person voice, slang, or expressions of personal opinion. \
Keep roughly the same length — do not expand short text into long paragraphs. \
Return only the rephrased text, no commentary.

Original:
{text}"""


async def _rephrase(text: str) -> str:
    """Rephrase text to remove personal voice. Returns original on failure."""
    if not text.strip() or not settings.deepseek_api_key:
        return text
    try:
        async with httpx.AsyncClient(timeout=25.0, trust_env=True) as client:
            resp = await client.post(
                "https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {settings.deepseek_api_key}"},
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": _REPHRASE_PROMPT.format(text=text)}],
                    "max_tokens": 600,
                    "temperature": 0.3,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning("Rephrase failed: %s", e)
        return text

# Approximate tokens per word (English)
TOKENS_PER_WORD = 1.3
MIN_CHUNK_WORDS = int(300 / TOKENS_PER_WORD)  # ~230 words
MAX_CHUNK_WORDS = int(500 / TOKENS_PER_WORD)  # ~385 words

METADATA_EXTRACTION_PROMPT = """\
You are analyzing a chunk of text from an investment analyst's document.
Extract the following metadata as JSON:

{{
  "tickers_mentioned": ["NVDA", "AAPL"],
  "temporal_scope": "short_term",
  "outlook_horizon": "3_month",
  "thesis_direction": "bullish",
  "chunk_type": "prediction"
}}

Fields:
- tickers_mentioned: stock ticker symbols (1-5 uppercase letters) found in text
- temporal_scope: "short_term" (< 6 months) / "long_term" (> 6 months) / "general" (philosophy)
- outlook_horizon: "1_month" / "3_month" / "6_month" / "multi_year"
- thesis_direction: "bullish" / "bearish" / "neutral" / "mixed"
- chunk_type: "philosophy" / "prediction" / "commentary" / "egf_explanation"

Rules:
- Only include tickers that are clearly stock symbols
- If no clear prediction, use "neutral" for thesis_direction
- If discussing general investing principles, use "philosophy" for chunk_type

Text to analyze:
---
{chunk_text}
---

Respond with ONLY the JSON object, no other text."""


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

    # Plain text / markdown
    return path.read_text(encoding="utf-8")


def chunk_text(text: str) -> list[str]:
    """
    Split text into chunks of 300–500 tokens at paragraph boundaries.
    Falls back to sentence boundaries if paragraphs are too large.
    """
    # Split on double newlines (paragraphs)
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())

        # If a single paragraph exceeds max, split it by sentences
        if para_words > MAX_CHUNK_WORDS:
            # Flush current
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_words = 0

            sentences = re.split(r"(?<=[.!?])\s+", para)
            sent_parts: list[str] = []
            sent_words = 0
            for sent in sentences:
                sw = len(sent.split())
                if sent_words + sw > MAX_CHUNK_WORDS and sent_parts:
                    chunks.append(" ".join(sent_parts))
                    sent_parts = []
                    sent_words = 0
                sent_parts.append(sent)
                sent_words += sw
            if sent_parts:
                chunks.append(" ".join(sent_parts))
            continue

        # Would adding this paragraph exceed max?
        if current_words + para_words > MAX_CHUNK_WORDS and current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = []
            current_words = 0

        current_parts.append(para)
        current_words += para_words

    # Flush remaining
    if current_parts:
        chunks.append("\n\n".join(current_parts))

    # Merge tiny chunks with their neighbors
    merged: list[str] = []
    for chunk in chunks:
        if merged and len(merged[-1].split()) < MIN_CHUNK_WORDS:
            merged[-1] = merged[-1] + "\n\n" + chunk
        else:
            merged.append(chunk)

    return merged


def _extract_metadata_sync(chunk_text_str: str) -> dict:
    """Synchronous metadata extraction via google-genai SDK (proxy-safe)."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key.strip())
    prompt = METADATA_EXTRACTION_PROMPT.format(chunk_text=chunk_text_str[:2000])

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=512,
            temperature=0.1,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    text = response.text.strip() if response.text else ""
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    return json.loads(text)


async def extract_metadata(chunk_text_str: str) -> dict:
    """Extract metadata from a chunk using Gemini Flash."""
    if not settings.gemini_api_key:
        logger.warning("No Gemini API key — returning default metadata")
        return _default_metadata()

    try:
        return await asyncio.to_thread(_extract_metadata_sync, chunk_text_str)
    except Exception:
        logger.exception("Metadata extraction failed, using defaults")
        return _default_metadata()


def _default_metadata() -> dict:
    return {
        "tickers_mentioned": [],
        "temporal_scope": "general",
        "outlook_horizon": "multi_year",
        "thesis_direction": "neutral",
        "chunk_type": "commentary",
    }


async def ingest_document(
    file_path: str | Path,
    db: AsyncSession,
    publish_date: date | None = None,
    progress_cb=None,
    r2_key: str | None = None,
) -> list[AnalystChunk]:
    """
    Full document ingestion pipeline.

    Args:
        file_path: Path to PDF or text file
        db: Async database session
        publish_date: When the document was published (for temporal tracking)
        progress_cb: Optional async callable(str) for streaming progress messages.
        r2_key: Optional R2 storage key to record on the UploadSource.

    Returns:
        List of created AnalystChunk records
    """
    async def _progress(msg: str) -> None:
        logger.info(msg)
        if progress_cb:
            await progress_cb(msg)

    # Instantiate here so runtime anonymization rules (loaded at startup) are included
    anonymizer = Anonymizer()

    path = Path(file_path)
    await _progress(f"Reading file: {path.name}")

    # 1. Read file
    raw_text = read_file(path)
    if not raw_text.strip():
        logger.warning("Empty document: %s", path.name)
        return []

    # 2. Chunk
    chunks = chunk_text(raw_text)
    total = len(chunks)
    await _progress(f"Split into {total} chunks.")

    # 3. Anonymize all chunks
    anonymized_chunks = []
    for chunk in chunks:
        result = anonymizer.scrub(chunk)
        anonymized_chunks.append(result.text)

    # 3b. Rephrase each chunk to remove personal voice / analyst identity
    rephrased_chunks = []
    for i, chunk in enumerate(anonymized_chunks):
        await _progress(f"Rephrasing chunk {i + 1}/{total}...")
        rephrased = await _rephrase(chunk)
        rephrased_chunks.append(rephrased)
        if i < len(anonymized_chunks) - 1:
            await asyncio.sleep(0.5)  # gentle rate limit on DeepSeek

    # 4. Extract metadata for each chunk (with small delay to avoid rate limiting)
    metadatas = []
    for i, chunk in enumerate(rephrased_chunks):
        await _progress(f"Extracting metadata for chunk {i + 1}/{total}...")
        meta = await extract_metadata(chunk)
        metadatas.append(meta)
        if i < len(rephrased_chunks) - 1:
            await asyncio.sleep(1.0)  # 1 req/sec stays under Flash rate limit

    # 5. Embed all chunks
    await _progress(f"Embedding {total} chunks...")
    embeddings = await embed_batch(rephrased_chunks)

    # 6. Create upload source
    upload = UploadSource(
        file_type=path.suffix.lstrip("."),
        r2_key=r2_key,
        extracted_json={
            "chunk_count": len(chunks),
            "file_name": path.name,
        },
    )
    db.add(upload)
    await db.flush()

    # 7. Store analyst chunks
    created_chunks: list[AnalystChunk] = []
    for i, (text, meta, emb) in enumerate(zip(rephrased_chunks, metadatas, embeddings)):
        # Determine primary ticker (first mentioned, if any)
        tickers = meta.get("tickers_mentioned", [])
        primary_ticker = tickers[0] if tickers else None

        chunk = AnalystChunk(
            upload_source_id=upload.id,
            ticker=primary_ticker,
            chunk_type=meta.get("chunk_type", "commentary"),
            content_text=text,
            embedding=emb,
            temporal_scope=meta.get("temporal_scope", "general"),
            metadata_json=meta,
            outlook_horizon=meta.get("outlook_horizon"),
            publish_date=publish_date,
            tickers_mentioned=tickers,
            thesis_direction=meta.get("thesis_direction"),
            retrieval_count=0,
            is_stale=False,
        )
        db.add(chunk)
        created_chunks.append(chunk)

    await db.flush()
    await _progress(f"Saved {len(created_chunks)} chunks to database.")

    return created_chunks
