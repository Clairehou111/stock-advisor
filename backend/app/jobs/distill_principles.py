"""
Principle distillation job — extracts investing principles from analyst chunks.

Triggered after document ingestion. For each chunk:
  1. Send to Gemini Flash with extraction prompt
  2. Compare extracted principles against existing derived_principles
  3. NEW → insert with confidence=0.3
  4. REINFORCED → bump confidence + times_stated
  5. CONTRADICTED → deactivate old, create new, link via superseded_by

Confidence formula: min(0.95, 0.3 + 0.15 * ln(times_stated))
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.tables import AnalystChunk, DerivedPrinciple

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """\
You are analyzing investment analyst text to extract investing principles.

A "principle" is a reusable investing rule, not a stock-specific prediction.
Examples of principles:
- "Accumulate during deep value zones using dollar-cost averaging"
- "PE expansion beyond historical range signals distribution zone"
- "Never sell at a loss in an AI bull market — trim into strength instead"

For each principle found, provide:
- text: the principle (one clear sentence)
- category: one of "valuation", "accumulation", "distribution", "risk", "sentiment"
- is_new_idea: true if this seems novel, false if it's a restatement of common investing wisdom

Return JSON array. If no principles found, return [].

Text:
---
{chunk_text}
---

Respond with ONLY the JSON array."""

COMPARISON_PROMPT = """\
Compare this new principle against existing principles. Is it:
1. NEW — a genuinely different investing idea
2. REINFORCED — same idea as an existing principle (even if worded differently)
3. CONTRADICTED — directly contradicts an existing principle

New principle: "{new_principle}"

Existing principles:
{existing_list}

Respond with JSON: {{"match_type": "NEW" | "REINFORCED" | "CONTRADICTED", "matched_id": "uuid or null", "reasoning": "brief explanation"}}
Only JSON, no other text."""


def _compute_confidence(times_stated: int) -> float:
    """Confidence formula: min(0.95, 0.3 + 0.15 * ln(times_stated))"""
    return min(0.95, 0.3 + 0.15 * math.log(max(1, times_stated)))


def _call_flash_once(prompt: str) -> str:
    """Single attempt at Gemini Flash."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key.strip())
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=1024,
            temperature=0.1,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    text = response.text.strip() if response.text else ""
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text


async def _call_flash_with_retry(prompt: str) -> str:
    """Gemini Flash with retry."""
    from app.llm.retry import retry
    return await retry(_call_flash_once, prompt, label="Gemini Flash/distill", sync=True)


async def _call_deepseek_once(prompt: str) -> str:
    """Single attempt at DeepSeek."""
    import httpx
    async with httpx.AsyncClient(timeout=30.0, trust_env=True) as client:
        resp = await client.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {settings.deepseek_api_key}"},
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "temperature": 0.1,
            },
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return text


async def _call_deepseek(prompt: str) -> str:
    """DeepSeek with retry."""
    from app.llm.retry import retry
    return await retry(_call_deepseek_once, prompt, label="DeepSeek/distill")


async def _call_flash(prompt: str) -> str:
    """Call Gemini Flash (retry) → fall back to DeepSeek (retry) → raise."""
    if settings.gemini_api_key:
        try:
            return await _call_flash_with_retry(prompt)
        except Exception as e:
            logger.warning("Gemini Flash exhausted retries (%s), falling back to DeepSeek", e)

    if settings.deepseek_api_key:
        return await _call_deepseek(prompt)

    raise RuntimeError("No LLM available — configure GEMINI_API_KEY or DEEPSEEK_API_KEY")


async def _extract_principles(chunk_text: str) -> list[dict]:
    """Extract candidate principles from a chunk."""
    prompt = EXTRACTION_PROMPT.format(chunk_text=chunk_text[:3000])
    try:
        result = await _call_flash(prompt)
        return json.loads(result)
    except Exception:
        logger.exception("Principle extraction failed")
        return []


async def _compare_principle(
    new_text: str,
    existing: list[DerivedPrinciple],
) -> dict:
    """Compare a new principle against existing ones."""
    if not existing:
        return {"match_type": "NEW", "matched_id": None}

    existing_list = "\n".join(
        f"- [{p.id}] {p.principle_text}" for p in existing[:30]
    )
    prompt = COMPARISON_PROMPT.format(
        new_principle=new_text,
        existing_list=existing_list,
    )

    try:
        result = await _call_flash(prompt)
        return json.loads(result)
    except Exception:
        logger.exception("Principle comparison failed, treating as NEW")
        return {"match_type": "NEW", "matched_id": None}


async def distill_from_chunks(
    chunk_ids: list[uuid.UUID],
    db: AsyncSession,
) -> int:
    """
    Run principle distillation on the given chunks.

    Returns number of principles created/updated.
    """
    # Load chunks
    result = await db.execute(
        select(AnalystChunk).where(AnalystChunk.id.in_(chunk_ids))
    )
    chunks = result.scalars().all()

    if not chunks:
        logger.warning("No chunks found for distillation")
        return 0

    # Load existing active principles
    existing_result = await db.execute(
        select(DerivedPrinciple).where(
            DerivedPrinciple.is_active == True  # noqa: E712
        )
    )
    existing_principles = list(existing_result.scalars().all())

    total_processed = 0
    now = datetime.now(timezone.utc)

    for chunk in chunks:
        candidates = await _extract_principles(chunk.content_text)

        for candidate in candidates:
            text = candidate.get("text", "").strip()
            category = candidate.get("category", "general")
            if not text:
                continue

            # Compare against existing
            comparison = await _compare_principle(text, existing_principles)
            match_type = comparison.get("match_type", "NEW")
            matched_id = comparison.get("matched_id")

            if match_type == "REINFORCED" and matched_id:
                # Find and update the matched principle
                matched = None
                for p in existing_principles:
                    if str(p.id) == str(matched_id):
                        matched = p
                        break

                if matched:
                    matched.times_stated += 1
                    matched.confidence_score = _compute_confidence(matched.times_stated)
                    matched.last_reinforced = now
                    # Add this chunk to source provenance
                    if matched.source_chunk_ids:
                        if chunk.id not in matched.source_chunk_ids:
                            matched.source_chunk_ids = matched.source_chunk_ids + [chunk.id]
                    else:
                        matched.source_chunk_ids = [chunk.id]
                    total_processed += 1
                    continue

            if match_type == "CONTRADICTED" and matched_id:
                # Deactivate old principle
                for p in existing_principles:
                    if str(p.id) == str(matched_id):
                        p.is_active = False
                        new_principle = DerivedPrinciple(
                            principle_text=text,
                            category=category,
                            confidence_score=0.3,
                            times_stated=1,
                            source_chunk_ids=[chunk.id],
                            first_seen=now,
                            last_reinforced=now,
                            is_active=True,
                        )
                        p.superseded_by = new_principle.id
                        db.add(new_principle)
                        existing_principles.append(new_principle)
                        total_processed += 1
                        break
                continue

            # NEW principle
            new_principle = DerivedPrinciple(
                principle_text=text,
                category=category,
                confidence_score=0.3,
                times_stated=1,
                source_chunk_ids=[chunk.id],
                first_seen=now,
                last_reinforced=now,
                is_active=True,
            )
            db.add(new_principle)
            existing_principles.append(new_principle)
            total_processed += 1

    await db.flush()
    logger.info("Principle distillation: processed %d principles from %d chunks", total_processed, len(chunks))
    return total_processed
