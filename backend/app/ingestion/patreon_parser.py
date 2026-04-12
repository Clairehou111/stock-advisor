"""
Patreon post ingestion pipeline.

Flow:
  1. Fetch post JSON via session cookie
  2. Extract text + images from ProseMirror content
  3. Filter identity/political nodes (pre-LLM, saves tokens)
  4. Single Gemini 3.1 Pro multimodal call: full text + all images → structured JSON
  5. Parse JSON into AnalystChunk rows
  6. Embed all chunks
  7. Upload images to R2 for backup
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import date, datetime, timezone

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.tables import AnalystChunk, UploadSource
from app.services.embedding_service import embed_batch

logger = logging.getLogger(__name__)


# ── Identity / political filtering ───────────────────────────────────────────

_BASE_IDENTITY_PATTERNS = [
    re.compile(r"https?://(?:www\.)?patreon\.com/\S+", re.I),
    re.compile(r"Copyright\s*©", re.I),
    re.compile(r"All rights reserved", re.I),
    re.compile(r"Disclaimer:", re.I),
    re.compile(r"personal financial advisor", re.I),
    re.compile(r"youtu\.be/|youtube\.com/watch", re.I),
    re.compile(r"Watch my (?:most recent|latest) video", re.I),
    re.compile(r"https?://\S+"),
]

_CACHED_IDENTITY_PATTERNS: list[re.Pattern] | None = None
_CACHED_POLITICAL_SIGNALS: list[str] | None = None


def _get_identity_patterns() -> list[re.Pattern]:
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


def _is_political(text: str) -> bool:
    signals = _get_political_signals()
    if not signals:
        return False
    t = text.lower()
    return sum(1 for kw in signals if kw in t) >= 2


def _is_identity(text: str) -> bool:
    return any(p.search(text) for p in _get_identity_patterns())


def _has_financial_content(text: str) -> bool:
    keywords = [
        "$", "%", "buy", "sell", "trim", "stock", "market", "s&p", "vix",
        "bear", "bull", "rally", "price", "support", "resistance", "pe",
        "earnings", "gdp", "inflation", "rate", "fed", "oil", "crypto",
        "nvda", "meta", "msft", "aapl", "tsla", "goog", "amzn",
    ]
    t = text.lower()
    return any(kw in t for kw in keywords)


# ── Content parsing ──────────────────────────────────────────────────────────


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


# ── Image download ───────────────────────────────────────────────────────────


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


# ── R2 upload ────────────────────────────────────────────────────────────────


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


# ── Gemini ingestion prompt ──────────────────────────────────────────────���───

_INGESTION_PROMPT = """\
Analyze this complete investment newsletter post. Return a JSON object with ALL information extracted.

POST TITLE: {title}

POST TEXT:
{text}

The images attached are financial charts referenced in the post, in order of appearance. \
Read the chart labels, axes, ticker symbols, and annotations carefully.

Return a JSON object:
{{
  "post_summary": "2-3 sentence summary of the post's overall thesis and key predictions",
  "chunks": [
    {{
      "section": "section title or topic",
      "content": "Rephrased in neutral third-person analyst language. Keep ALL price levels, \
percentages, timeframes, ticker symbols, and directional signals. Remove first-person voice, \
slang, personal opinion. Include chart analysis inline where the chart relates to this section. \
Do NOT over-summarize — preserve the detail and reasoning.",
      "primary_ticker": "^GSPC or NVDA or null if multi-topic",
      "tickers_mentioned": ["^GSPC", "NVDA"],
      "chunk_type": "prediction|philosophy|commentary|egf_explanation",
      "temporal_scope": "short_term|long_term|general",
      "thesis_direction": "bullish|bearish|neutral|mixed",
      "key_levels": [
        {{"price": 6780, "type": "resistance|support|buy|sell|target", \
"significance": "minor|major|critical", "note": "optional short note"}}
      ]
    }}
  ]
}}

RULES:
- Use REAL tradeable ticker symbols. Read chart labels to identify correct tickers.
- For indices: ^GSPC for S&P 500/SPX, ^IXIC for NASDAQ, ^DJI for Dow, ^VIX for VIX
- For commodities/crypto: GLD for gold, USO for oil, BTC-USD for Bitcoin, ETH-USD for Ethereum, \
SOL-USD for Solana
- If a chart shows a company name (e.g. "Circle Internet Group"), find its real ticker symbol
- Include index as primary_ticker when section discusses its levels/forecast
- When a section discusses multiple stocks with SPECIFIC price levels or targets for each, \
create a SEPARATE chunk per stock. Keep a parent chunk for the portfolio-level strategy.
- Keep historical data tables (e.g. drawdown percentages) as their own chunk — do NOT merge \
or drop them
- Each meaningful topic = separate chunk
- Merge chart analysis into relevant chunk content — do not create separate chart-only chunks
- Remove author names, personal website URLs, identifying information
- Preserve full analytical detail — do not compress multi-paragraph analysis into one sentence
- key_levels: extract ALL specific price levels mentioned with their significance
- Return ONLY valid JSON, no markdown fences"""


# ── Single Gemini call ───────────────────────────────────────────────────────


async def _call_gemini_ingest(
    text: str, title: str, images: list[bytes], progress_cb=None,
) -> dict:
    """Single Gemini 3.1 Pro multimodal call: text + images → structured JSON."""
    from google import genai
    from google.genai import types
    from app.llm.retry import retry

    async def _log(msg: str) -> None:
        logger.info(msg)
        if progress_cb:
            await progress_cb(msg)

    prompt = _INGESTION_PROMPT.format(title=title, text=text)

    parts = [prompt]
    for img_bytes in images:
        parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))

    await _log(f"Calling Gemini 3.1 Pro with {len(images)} images...")

    def _call():
        client = genai.Client(api_key=settings.gemini_api_key.strip())
        response = client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=parts,
            config=types.GenerateContentConfig(
                max_output_tokens=65536,
                temperature=0.1,
            ),
        )
        return response.text.strip()

    raw = await retry(_call, max_retries=3, label="Gemini/ingestion", sync=True)

    # Strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    result = json.loads(raw)
    chunk_count = len(result.get("chunks", []))
    await _log(f"Gemini returned {chunk_count} chunks")
    return result


# ── Patreon API ──────────────────────────────────────────────────────────────


async def fetch_post_json(post_id: str) -> dict:
    """Fetch post content JSON from Patreon using session cookie."""
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


# ── Main pipeline ────────────────────────────────────────────────────────────


async def ingest_patreon_post(
    post_id: str,
    db: AsyncSession,
    progress_cb=None,
    force: bool = False,
) -> dict:
    """
    Full Patreon post ingestion pipeline.

    Single Gemini 3.1 Pro call processes the entire post (text + images).
    """
    async def _log(msg: str) -> None:
        logger.info(msg)
        if progress_cb:
            await progress_cb(msg)

    await _log(f"Fetching post {post_id}...")

    # 1. Fetch
    data = await fetch_post_json(post_id)
    attrs = data.get("data", {}).get("attributes", {})
    title = (attrs.get("title") or "").strip() or f"Post {post_id}"
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

    # 2. Find or create UploadSource
    from sqlalchemy import delete, select as sa_select, func
    existing_result = await db.execute(
        sa_select(UploadSource)
        .where(UploadSource.extracted_json["post_id"].astext == post_id)
        .order_by(UploadSource.upload_timestamp.desc())
    )
    all_uploads = existing_result.scalars().all()

    upload = all_uploads[0] if all_uploads else None
    for old in all_uploads[1:]:
        await db.execute(delete(AnalystChunk).where(AnalystChunk.upload_source_id == old.id))
        await db.delete(old)
    await db.flush()

    if force and upload:
        await _log("Force re-ingest — wiping existing chunks")
        await db.execute(delete(AnalystChunk).where(AnalystChunk.upload_source_id == upload.id))
        await db.flush()

    if not upload:
        upload = UploadSource(
            file_type="patreon",
            r2_key=f"patreon/posts/{post_id}/content.json",
            extracted_json={"post_id": post_id, "title": title},
            raw_content=content_json,
        )
        db.add(upload)
        await db.flush()
    elif upload.extracted_json:
        upload.extracted_json = {**upload.extracted_json, "title": title}

    # 3. Filter identity/political nodes
    filtered_nodes = _filter_nodes(all_nodes)
    removed = len(all_nodes) - len(filtered_nodes)
    await _log(f"Filtered {removed} identity/political nodes, {len(filtered_nodes)} remain")

    # 4. Extract text and image URLs
    text_nodes = [n for n in filtered_nodes if n["type"] == "text"]
    image_nodes = [n for n in filtered_nodes if n["type"] == "image"]
    all_text = "\n\n".join(n["text"] for n in text_nodes if n["text"].strip())
    image_urls = [n.get("attrs", {}).get("src", "") for n in image_nodes]
    image_urls = [u for u in image_urls if u]

    await _log(f"Text: {len(all_text):,} chars, {len(image_urls)} images to download")

    # 5. Download ALL images (no cap)
    images: list[bytes] = []
    for url in image_urls:
        img = await _download_image(url)
        if img:
            images.append(img)
    await _log(f"Downloaded {len(images)}/{len(image_urls)} images")

    # 6. Single Gemini 3.1 Pro call
    result = await _call_gemini_ingest(all_text, title, images, progress_cb=progress_cb)

    # 7. Parse chunks into AnalystChunk rows
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
                "post_id": post_id,
                "key_levels": c.get("key_levels", []),
            },
        )
        db.add(chunk)
        created_chunks.append(chunk)

    await _log(f"Created {len(created_chunks)} chunks")

    # 8. Embed all chunks
    if created_chunks:
        await _log(f"Embedding {len(created_chunks)} chunks...")
        texts = [c.content_text for c in created_chunks]
        embeddings = await embed_batch(texts)
        for chunk, emb in zip(created_chunks, embeddings):
            chunk.embedding = emb

    await db.flush()

    # 9. Upload images to R2 in background (non-blocking)
    if images and all([settings.r2_endpoint_url, settings.r2_access_key_id,
                       settings.r2_secret_access_key, settings.r2_bucket_name]):
        await _log("Uploading images to R2...")
        for i, img_bytes in enumerate(images):
            try:
                r2_key = f"patreon/posts/{post_id}/charts/{i}.jpg"
                await asyncio.to_thread(_upload_to_r2, img_bytes, r2_key)
            except Exception as e:
                logger.warning("R2 upload failed for chart %d: %s", i, e)

    # 10. Count total chunks (including any from previous partial runs)
    total_chunks_q = await db.execute(
        sa_select(func.count(AnalystChunk.id)).where(AnalystChunk.upload_source_id == upload.id)
    )
    total_chunks = total_chunks_q.scalar() or len(created_chunks)

    image_count = len(images)
    await _log(f"Done: {total_chunks} chunks, {image_count} charts")

    return {
        "chunk_count": total_chunks,
        "signal_count": 0,  # signals now embedded in chunks as key_levels
        "image_count": image_count,
        "upload_source_id": str(upload.id),
        "title": title,
    }
