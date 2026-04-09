"""
Patreon post ingestion pipeline.

Flow:
  1. Fetch post JSON via session cookie
  2. Save raw content_json_string → upload_sources.raw_content
  3. Strip identity markers (name, URLs, copyright) and political paragraphs
  4. Extract structured trade signals via LLM
  5. Split into sections using CONTENTS headings
  6. Per section:
     - Rephrase text → neutral analyst voice
     - Download images → R2, analyze via Gemini Vision
     - Merge text + chart descriptions into section chunk
  7. Embed + store analyst_chunks
  8. Distill principles
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import tempfile
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.tables import AnalystChunk, TradeSignal, UploadSource
from app.services.embedding_service import embed_batch

logger = logging.getLogger(__name__)

# Generic identity patterns safe to keep in source — no identifying information.
# Sensitive patterns (specific domain names, company names) come from
# settings.identity_strip_patterns (env var), loaded at call time.
_BASE_IDENTITY_PATTERNS = [
    re.compile(r"https?://(?:www\.)?patreon\.com/\S+", re.I),
    re.compile(r"Copyright\s*©", re.I),
    re.compile(r"All rights reserved", re.I),
    re.compile(r"Disclaimer:", re.I),
    re.compile(r"personal financial advisor", re.I),
    re.compile(r"youtu\.be/|youtube\.com/watch", re.I),
    re.compile(r"Watch my (?:most recent|latest) video", re.I),
    re.compile(r"https?://\S+"),  # any remaining bare URLs
]


_CACHED_IDENTITY_PATTERNS: list[re.Pattern] | None = None
_CACHED_POLITICAL_SIGNALS: list[str] | None = None


def _get_identity_patterns() -> list[re.Pattern]:
    """Merge base patterns with extra patterns from env var. Parsed once and cached."""
    global _CACHED_IDENTITY_PATTERNS
    if _CACHED_IDENTITY_PATTERNS is not None:
        return _CACHED_IDENTITY_PATTERNS
    patterns = list(_BASE_IDENTITY_PATTERNS)
    if settings.identity_strip_patterns:
        try:
            extras = json.loads(settings.identity_strip_patterns)
            for pat in extras:
                patterns.append(re.compile(pat, re.I))
        except Exception:
            logger.warning("IDENTITY_STRIP_PATTERNS is not valid JSON — using base patterns only")
    _CACHED_IDENTITY_PATTERNS = patterns
    return patterns


def _get_political_signals() -> list[str]:
    """Return political signal keywords from env var. Parsed once and cached."""
    global _CACHED_POLITICAL_SIGNALS
    if _CACHED_POLITICAL_SIGNALS is not None:
        return _CACHED_POLITICAL_SIGNALS
    if not settings.political_signals:
        _CACHED_POLITICAL_SIGNALS = []
        return _CACHED_POLITICAL_SIGNALS
    try:
        _CACHED_POLITICAL_SIGNALS = [s.lower() for s in json.loads(settings.political_signals)]
    except Exception:
        logger.warning("POLITICAL_SIGNALS is not valid JSON — political filtering disabled")
        _CACHED_POLITICAL_SIGNALS = []
    return _CACHED_POLITICAL_SIGNALS

_REPHRASE_PROMPT = """\
Rephrase the following investment analysis text in neutral, third-person language \
as a generic quantitative analyst would write it. \
Keep all tickers, price levels, percentage moves, timeframes, and directional signals intact. \
Remove any first-person voice, slang, or expressions of personal opinion. \
Keep roughly the same length — do not expand short text into long paragraphs. \
Return only the rephrased text, no commentary.

Original:
{text}"""

_SIGNAL_EXTRACTION_PROMPT = """\
Extract all explicit buy, trim, sell, or watch signals from the text below.
Return a JSON array. Each item: {{"ticker": "NVDA", "action": "buy", "price_level": 522.0, "confidence": "high", "context": "buying META at $522"}}
- action: "buy" / "trim" / "sell" / "watch"
- price_level: numeric if explicitly stated, null if not
- confidence: "high" if analyst says they are actively doing it, "medium" if suggested, "low" if speculative
- context: the exact short phrase it came from (max 100 chars)
If no signals, return [].
Return only the JSON array.

Text:
{text}"""

_CHART_VISION_PROMPT = """\
Describe the financial chart in this image. Extract:
- Ticker symbol if shown (keep this)
- Timeframe (daily, weekly, monthly, etc.)
- Current price level if annotated
- Key support and resistance levels with prices
- Trend direction (bullish/bearish/sideways)
- Any annotated price targets or forecasts
- Indicator readings if shown (RSI, MACD, etc.)
Ignore and do not transcribe any watermarks, website URLs, author names, or copyright notices.
Be concise. If this is not a financial chart, say "Non-chart image"."""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_political(text: str) -> bool:
    signals = _get_political_signals()
    if not signals:
        return False
    t = text.lower()
    return sum(1 for kw in signals if kw in t) >= 2


def _is_identity(text: str) -> bool:
    return any(p.search(text) for p in _get_identity_patterns())


def _has_financial_content(text: str) -> bool:
    """Return True if paragraph contains market/financial signal worth keeping."""
    keywords = [
        "$", "%", "buy", "sell", "trim", "stock", "market", "s&p", "vix",
        "bear", "bull", "rally", "price", "support", "resistance", "pe",
        "earnings", "gdp", "inflation", "rate", "fed", "oil", "crypto",
        "nvda", "meta", "msft", "aapl", "tsla", "goog", "amzn",
    ]
    t = text.lower()
    return any(kw in t for kw in keywords)


async def _deepseek_call(prompt: str, max_tokens: int = 300, temperature: float = 0.3) -> str:
    """Shared DeepSeek call with retry. Raises on exhaustion — callers must handle."""
    from app.llm.retry import retry

    async def _call():
        async with httpx.AsyncClient(timeout=30.0, trust_env=True) as client:
            resp = await client.post(
                "https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {settings.deepseek_api_key}"},
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

    return await retry(_call, label="DeepSeek/ingestion")


async def _rephrase(text: str) -> str:
    if not text.strip() or not settings.deepseek_api_key:
        return text
    try:
        result = await _deepseek_call(_REPHRASE_PROMPT.format(text=text), max_tokens=300, temperature=0.3)
        return result or text
    except Exception as e:
        logger.warning("Rephrase failed after retries: %s", e)
        return text


async def _extract_signals_chunk(text: str) -> list[dict]:
    """Extract signals from a single text chunk (max ~6000 chars)."""
    if not settings.deepseek_api_key or not text.strip():
        return []
    try:
        raw = await _deepseek_call(
            _SIGNAL_EXTRACTION_PROMPT.format(text=text[:6000]),
            max_tokens=600, temperature=0.1,
        )
        if not raw:
            return []
        # Strip markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()
        if not raw or raw == "[]":
            return []
        # Try to salvage truncated JSON arrays
        if raw.startswith("[") and not raw.endswith("]"):
            raw = raw.rsplit("}", 1)[0] + "}]"
        return json.loads(raw)
    except Exception as e:
        logger.warning("Signal extraction chunk failed: %s", e)
        return []


async def _extract_signals(
    sections: list[dict],
    publish_date: date | None,
    progress_cb=None,
) -> list[dict]:
    """Extract trade signals section by section to avoid context overload."""
    all_signals: list[dict] = []
    total = len(sections)
    for i, section in enumerate(sections, 1):
        title = section.get('title', '?')
        if progress_cb:
            await progress_cb(f"Extracting signals {i}/{total}: {title}")
        text = " ".join(n["text"] for n in section["nodes"] if n["type"] == "text")
        if not text.strip():
            continue
        signals = await _extract_signals_chunk(text)
        all_signals.extend(signals)
        if signals:
            await asyncio.sleep(0.5)
    return all_signals


def _analyze_chart_once(image_bytes: bytes) -> str:
    """Single attempt at Gemini Vision chart analysis."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key.strip())

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            _CHART_VISION_PROMPT,
        ],
        config=types.GenerateContentConfig(
            max_output_tokens=512,
            temperature=0.1,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text.strip() if response.text else "Chart analysis unavailable."


async def _analyze_chart(image_bytes: bytes) -> str:
    """Chart analysis with retry."""
    from app.llm.retry import retry
    try:
        return await retry(_analyze_chart_once, image_bytes, label="Gemini Vision", sync=True)
    except Exception as e:
        logger.warning("Chart analysis failed after retries: %s", e)
        return "Chart analysis unavailable — service temporarily overloaded."


async def _download_image(url: str) -> bytes | None:
    """Download image from Patreon CDN using session cookie."""
    try:
        async with httpx.AsyncClient(timeout=30.0, trust_env=True) as client:
            resp = await client.get(
                url,
                headers={
                    "referer": "https://www.patreon.com/",
                    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                },
                cookies={"session_id": settings.patreon_session_id} if settings.patreon_session_id else {},
                follow_redirects=True,
            )
            resp.raise_for_status()
            return resp.content
    except Exception as e:
        logger.warning("Image download failed %s: %s", url[:60], e)
        return None


def _r2_key_exists(r2_key: str) -> bool:
    """Return True if the key already exists in R2."""
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError

    s3 = boto3.client(
        "s3",
        endpoint_url=settings.r2_endpoint_url,
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )
    try:
        s3.head_object(Bucket=settings.r2_bucket_name, Key=r2_key)
        return True
    except ClientError:
        return False


def _upload_to_r2(image_bytes: bytes, r2_key: str) -> str:
    """Upload image to R2 and return the key."""
    import boto3
    from botocore.config import Config

    s3 = boto3.client(
        "s3",
        endpoint_url=settings.r2_endpoint_url,
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )
    s3.put_object(
        Bucket=settings.r2_bucket_name,
        Key=r2_key,
        Body=image_bytes,
        ContentType="image/jpeg",
    )
    return r2_key


# ── Content parsing ────────────────────────────────────────────────────────────

def _extract_nodes(content_json: dict) -> list[dict]:
    """Flatten ProseMirror document into ordered list of text/image nodes."""
    nodes = []

    def walk(node: dict) -> None:
        ntype = node.get("type", "")
        if ntype == "image":
            nodes.append({"type": "image", "attrs": node.get("attrs", {})})
        elif ntype in ("paragraph", "listItem"):
            texts = [
                c.get("text", "")
                for c in node.get("content", [])
                if c.get("type") == "text"
            ]
            combined = " ".join(texts).strip()
            if combined:
                nodes.append({"type": "text", "text": combined})
        for child in node.get("content", []):
            walk(child)

    for top in content_json.get("content", []):
        walk(top)

    return nodes


def _split_into_sections(nodes: list[dict]) -> list[dict]:
    """
    Split nodes into sections using CONTENTS headings as boundaries.
    Each section: {"title": str, "nodes": [...]}
    Falls back to a single section if no CONTENTS structure found.
    """
    # Find CONTENTS node and the lines after it (section titles)
    contents_idx = None
    for i, node in enumerate(nodes):
        if node["type"] == "text" and node["text"].strip().upper() == "CONTENTS":
            contents_idx = i
            break

    if contents_idx is None:
        return [{"title": "Main", "nodes": nodes}]

    # Collect section titles from lines immediately after CONTENTS
    section_titles = []
    i = contents_idx + 1
    while i < len(nodes) and nodes[i]["type"] == "text":
        t = nodes[i]["text"].strip()
        if t and len(t) < 120:
            section_titles.append(t)
        i += 1
        if len(section_titles) > 20:
            break

    if not section_titles:
        return [{"title": "Main", "nodes": nodes}]

    # Split document at each section title occurrence
    sections = []
    current_title = "Preamble"
    current_nodes: list[dict] = []

    for node in nodes:
        if node["type"] == "text":
            matched = next((t for t in section_titles if node["text"].strip() == t), None)
            if matched:
                if current_nodes:
                    sections.append({"title": current_title, "nodes": current_nodes})
                current_title = matched
                current_nodes = []
                continue
        current_nodes.append(node)

    if current_nodes:
        sections.append({"title": current_title, "nodes": current_nodes})

    return sections if sections else [{"title": "Main", "nodes": nodes}]


def _filter_nodes(nodes: list[dict]) -> list[dict]:
    """Remove identity markers, political-only paragraphs, and bare URLs."""
    filtered = []
    for node in nodes:
        if node["type"] == "image":
            filtered.append(node)
            continue
        text = node["text"]
        if _is_identity(text):
            continue
        if _is_political(text) and not _has_financial_content(text):
            continue
        filtered.append(node)
    return filtered


# ── Main pipeline ──────────────────────────────────────────────────────────────

async def fetch_post_json(post_id: str) -> dict:
    """Fetch post content JSON from Patreon using session cookie."""
    # Use direct post endpoint
    post_url = (
        f"https://www.patreon.com/api/posts/{post_id}"
        f"?fields%5Bpost%5D=title%2Ccontent_json_string%2Cpublished_at%2Curl"
        f"&json-api-version=1.0"
    )
    headers = {
        "content-type": "application/vnd.api+json",
        "referer": f"https://www.patreon.com/posts/{post_id}",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
    }
    cookies = {}
    if settings.patreon_session_id:
        cookies["session_id"] = settings.patreon_session_id

    async with httpx.AsyncClient(timeout=30.0, trust_env=True, cookies=cookies) as client:
        resp = await client.get(post_url, headers=headers)
        resp.raise_for_status()
        return resp.json()


async def ingest_patreon_post(
    post_id: str,
    db: AsyncSession,
    progress_cb=None,
    force: bool = False,
) -> dict:
    """
    Full Patreon post ingestion pipeline.

    Args:
        post_id: Patreon post ID (numeric string or extracted from URL)
        db: Async database session
        progress_cb: Optional async callback(message: str) for UI progress updates

    Returns:
        Summary dict with chunk_count, signal_count, image_count
    """
    async def _log(msg: str) -> None:
        logger.info(msg)
        if progress_cb:
            await progress_cb(msg)

    await _log(f"Fetching post {post_id}...")

    # 1. Fetch
    data = await fetch_post_json(post_id)
    attrs = data.get("data", {}).get("attributes", {})
    title = attrs.get("title", f"Post {post_id}")
    published_at_str = attrs.get("published_at", "")
    cjs = attrs.get("content_json_string", "")

    if not cjs:
        raise ValueError(f"No content_json_string in post {post_id}")

    publish_date: date | None = None
    if published_at_str:
        try:
            publish_date = datetime.fromisoformat(published_at_str.replace("Z", "+00:00")).date()
        except Exception:
            pass

    await _log(f"Post: '{title}' ({publish_date}) — {len(cjs):,} chars")

    content_json = json.loads(cjs)
    all_nodes = _extract_nodes(content_json)
    await _log(f"Extracted {len(all_nodes)} nodes ({sum(1 for n in all_nodes if n['type'] == 'image')} images)")

    # 2. Find existing upload source or create fresh
    from sqlalchemy import delete, select as sa_select, func
    existing_result = await db.execute(
        sa_select(UploadSource)
        .where(UploadSource.extracted_json["post_id"].astext == post_id)
        .order_by(UploadSource.upload_timestamp.desc())
    )
    all_uploads = existing_result.scalars().all()

    # Keep the most recent, delete any duplicates from previous failed runs
    upload = all_uploads[0] if all_uploads else None
    for old in all_uploads[1:]:
        await db.execute(delete(AnalystChunk).where(AnalystChunk.upload_source_id == old.id))
        await db.execute(delete(TradeSignal).where(TradeSignal.upload_source_id == old.id))
        await db.delete(old)
    if len(all_uploads) > 1:
        await db.flush()
        await _log(f"Cleaned up {len(all_uploads) - 1} duplicate upload source(s)")

    if upload and force:
        # Force re-ingest: wipe everything and start clean
        await db.execute(delete(AnalystChunk).where(AnalystChunk.upload_source_id == upload.id))
        await db.execute(delete(TradeSignal).where(TradeSignal.upload_source_id == upload.id))
        await db.delete(upload)
        await db.flush()
        upload = None
        done_sections: set[str] = set()
        await _log("Force re-ingest — cleared all existing data")

    if upload and not force:
        # Resume: discover which sections already have chunks
        done_q = await db.execute(
            sa_select(AnalystChunk.metadata_json["section"].astext)
            .where(AnalystChunk.upload_source_id == upload.id)
            .distinct()
        )
        done_sections = {row[0] for row in done_q.all() if row[0]}

        # Always re-extract signals (cheap, may have improved)
        await db.execute(delete(TradeSignal).where(TradeSignal.upload_source_id == upload.id))
        await db.flush()
        await _log(f"Resuming — {len(done_sections)} sections already processed: {', '.join(sorted(done_sections)) or 'none'}")

    if not upload:
        done_sections = set()
        upload = UploadSource(
            file_type="patreon",
            r2_key=f"patreon/posts/{post_id}/content.json",
            extracted_json={"post_id": post_id, "title": title},
            raw_content=content_json,
        )
        db.add(upload)
        await db.flush()
        if not force:
            await _log("Fresh ingest — no prior data found")

    # 3. Filter identity/political nodes
    filtered_nodes = _filter_nodes(all_nodes)
    removed = len(all_nodes) - len(filtered_nodes)
    await _log(f"Filtered {removed} identity/political nodes, {len(filtered_nodes)} remain")

    # 5. Split into sections (needed for signal extraction too)
    sections = _split_into_sections(filtered_nodes)
    await _log(f"Split into {len(sections)} sections")

    # 4. Extract trade signals per section
    await _log("Extracting trade signals...")
    raw_signals = await _extract_signals(sections, publish_date, progress_cb=_log)

    signal_count = 0
    for sig in raw_signals:
        ticker = (sig.get("ticker") or "").upper().strip()
        action = (sig.get("action") or "").lower().strip()
        if not ticker or action not in ("buy", "trim", "sell", "watch"):
            continue
        db.add(TradeSignal(
            ticker=ticker,
            action=action,
            price_level=sig.get("price_level"),
            confidence=sig.get("confidence"),
            post_id=post_id,
            publish_date=publish_date,
            context_text=sig.get("context", "")[:500],
            upload_source_id=upload.id,
        ))
        signal_count += 1
    await _log(f"Extracted {signal_count} trade signals")

    # 6. Process each section
    created_chunks: list[AnalystChunk] = []
    image_count = 0

    for sec_idx, section in enumerate(sections):
        sec_title = section["title"]
        sec_nodes = section["nodes"]

        if sec_title in done_sections:
            await _log(f"  Skipping '{sec_title}' (already done)")
            continue

        try:
            await _process_section(
                sec_title, sec_nodes, post_id, publish_date, upload,
                created_chunks, db, _log, _download_image, _analyze_chart,
            )
            image_count += sum(1 for n in sec_nodes if n["type"] == "image")
        except Exception as e:
            logger.exception("Section '%s' failed, skipping", sec_title)
            await _log(f"  WARNING: Section '{sec_title}' failed ({e}), skipping")

    # 7. Embed all chunks in batches
    await _log(f"Embedding {len(created_chunks)} chunks...")
    texts = [c.content_text for c in created_chunks]
    if texts:
        embeddings = await embed_batch(texts)
        for chunk, emb in zip(created_chunks, embeddings):
            chunk.embedding = emb

    await db.flush()

    # Total chunk count includes previously stored chunks (resume case)
    total_chunks_q = await db.execute(
        sa_select(func.count(AnalystChunk.id)).where(AnalystChunk.upload_source_id == upload.id)
    )
    total_chunks = total_chunks_q.scalar() or len(created_chunks)
    await _log(f"Done: {total_chunks} chunks total ({len(created_chunks)} new), {signal_count} signals, {image_count} charts")

    return {
        "chunk_count": total_chunks,
        "signal_count": signal_count,
        "image_count": image_count,
        "upload_source_id": str(upload.id),
        "title": title,
    }
