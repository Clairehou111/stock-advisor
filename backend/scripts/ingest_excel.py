"""
CLI: Load Sid Sloth's Excel workbook into the database.

Usage:
    python -m scripts.ingest_excel <path-to-xlsx>

Flow:
    1. parse_workbook() → stocks, principles, diffs
    2. Anonymize all text fields
    3. Upsert stock_predictions (mark old as not current)
    4. Upsert principle_corpus
    5. Create upload_sources record
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import asdict

import httpx
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.session import async_session
from app.ingestion.anonymizer import Anonymizer
from app.ingestion.excel_parser import parse_workbook
from app.models.tables import PrincipleCorpus, StockPrediction, UploadSource

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_REPHRASE_PROMPT = """\
Rephrase the following stock analysis note in neutral, third-person language \
as if written by a generic quantitative analyst. \
Keep all numbers, price levels, and signals intact. \
Remove any distinctive personal voice, slang, or first-person expressions. \
Keep the output roughly the same length as the input — do NOT expand short phrases into long paragraphs. \
Return only the rephrased text, no extra commentary.

Original:
{text}"""


async def _rephrase(text: str) -> str:
    """Rephrase text via DeepSeek to remove distinctive authorial voice."""
    if not text or not text.strip():
        return text
    api_key = settings.deepseek_api_key
    if not api_key:
        return text  # no-op if key missing

    try:
        async with httpx.AsyncClient(timeout=20.0, trust_env=False) as client:
            resp = await client.post(
                "https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": _REPHRASE_PROMPT.format(text=text)}],
                    "max_tokens": 200,
                    "temperature": 0.3,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning("Rephrase failed (%s), using original", e)
        return text


async def ingest(
    file_path: str,
    return_result: bool = False,
    db=None,
    progress_cb=None,
    r2_key: str | None = None,
) -> dict | None:
    """
    Ingest Excel workbook.
    - When called from CLI: db=None, opens its own session, exits on error.
    - When called from API: pass db session, returns result dict.
    - progress_cb: optional async callable(str) for streaming progress messages.
    - r2_key: optional R2 storage key to record on the UploadSource.
    """
    async def _progress(msg: str) -> None:
        logger.info(msg)
        if progress_cb:
            await progress_cb(msg)

    await _progress("Parsing workbook...")
    wb_data = parse_workbook(file_path)

    current = wb_data["current"]
    if not current or not current.stocks:
        if return_result:
            raise ValueError("No stocks found in workbook")
        logger.error("No stocks found in current sheet")
        sys.exit(1)

    principles = wb_data["principles"]
    diffs = wb_data["diffs"]
    total_stocks = len(current.stocks)
    await _progress(f"Found {total_stocks} stocks and {len(principles)} principles.")

    async def _run(db: AsyncSession) -> dict:
        # Instantiate here so runtime anonymization rules (loaded at startup) are included
        anonymizer = Anonymizer()
        # Create upload source record
        upload = UploadSource(
            file_type="xlsx",
            r2_key=r2_key,
            sheet_name=current.sheet_name,
            extracted_json={
                "stock_count": len(current.stocks),
                "principle_count": len(principles),
                "snapshot_count": len(wb_data["dated_snapshots"]),
            },
            change_summary=_summarize_diffs(diffs),
        )
        db.add(upload)
        await db.flush()

        # Mark all existing predictions as not current
        await db.execute(
            update(StockPrediction)
            .where(StockPrediction.is_current == True)  # noqa: E712
            .values(is_current=False)
        )

        # Insert new stock predictions
        inserted = 0
        for i, stock in enumerate(current.stocks, 1):
            stock_dict = asdict(stock)
            await _progress(f"Processing stock {i}/{total_stocks}: {stock_dict['ticker']}")

            # Anonymize text fields
            for field in ("stock_name", "strategy_text", "trend_status"):
                if stock_dict.get(field):
                    stock_dict[field] = anonymizer.scrub(stock_dict[field]).text

            # Rephrase personality-heavy fields to remove distinctive voice
            for field in ("strategy_text", "trend_status"):
                if stock_dict.get(field):
                    stock_dict[field] = await _rephrase(stock_dict[field])
                    await asyncio.sleep(0.5)  # rate limit

            pred = StockPrediction(
                ticker=stock_dict["ticker"],
                stock_name=stock_dict["stock_name"],
                buy_high=stock_dict["buy_high"],
                buy_low=stock_dict["buy_low"],
                sell_start=stock_dict["sell_start"],
                pe_range_high=stock_dict["pe_range_high"],
                pe_range_low=stock_dict["pe_range_low"],
                fair_value=stock_dict["fair_value"],
                egf=stock_dict["egf"],
                egf_direction=stock_dict["egf_direction"],
                egf_12m=stock_dict["egf_12m"],
                fundamentals=stock_dict["fundamentals"],
                trend_status=stock_dict["trend_status"],
                prob_new_ath=stock_dict["prob_new_ath"],
                strategy_text=stock_dict["strategy_text"],
                analyst_labels={"category": stock_dict["category"]},
                upload_source_id=upload.id,
                is_current=True,
            )
            db.add(pred)
            inserted += 1

        await _progress(f"Saved {inserted} stocks to database.")

        # Upsert principles
        principle_count = 0
        total_principles = len(principles)
        for i, p in enumerate(principles, 1):
            await _progress(f"Processing principle {i}/{total_principles}: {p.title[:50]}")
            content = anonymizer.scrub(p.content).text
            title = anonymizer.scrub(p.title).text
            # Rephrase to neutral voice
            content = await _rephrase(content)
            await asyncio.sleep(0.5)

            # Check if principle already exists (by section number + title)
            existing = await db.execute(
                select(PrincipleCorpus).where(
                    PrincipleCorpus.principle_text.ilike(f"%{title[:50]}%")
                ).limit(1)
            )
            existing_row = existing.scalars().first()

            if existing_row:
                existing_row.principle_text = f"**{p.section_number}. {title}**\n{content}"
                existing_row.category = p.category
                existing_row.version = existing_row.version + 1
            else:
                db.add(PrincipleCorpus(
                    principle_text=f"**{p.section_number}. {title}**\n{content}",
                    category=p.category,
                    source_ids=[upload.id],
                ))
            principle_count += 1

        logger.info("Ingestion complete. Upload source ID: %s", upload.id)

        return {
            "stock_count": inserted,
            "principle_count": principle_count,
            "upload_source_id": str(upload.id),
        }

    if db is not None:
        # Called from API — use provided session, caller handles commit
        return await _run(db)

    # Called from CLI — open own session and commit
    async with async_session() as session:
        result = await _run(session)
        await session.commit()
    if not return_result:
        return None
    return result


def _summarize_diffs(diffs: list[dict]) -> str:
    if not diffs:
        return "No previous snapshots to diff against."

    parts = []
    for diff in diffs:
        changes = diff["changes"]
        added = sum(1 for c in changes if c["change"] == "added")
        removed = sum(1 for c in changes if c["change"] == "removed")
        updated = sum(1 for c in changes if c["change"] == "updated")
        parts.append(
            f"{diff['from']} → {diff['to']}: "
            f"{added} added, {removed} removed, {updated} updated"
        )
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Ingest Sid Sloth Excel workbook into DB")
    parser.add_argument("file_path", help="Path to the .xlsx file")
    args = parser.parse_args()

    asyncio.run(ingest(args.file_path))


if __name__ == "__main__":
    main()
