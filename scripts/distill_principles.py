"""
CLI: Run principle distillation on all non-stale analyst chunks.

Usage:
    python -m scripts.distill_principles [--limit N]
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from sqlalchemy import select

from app.db.session import async_session
from app.jobs.distill_principles import distill_from_chunks
from app.models.tables import AnalystChunk

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def main(limit: int) -> None:
    async with async_session() as db:
        result = await db.execute(
            select(AnalystChunk.id)
            .where(AnalystChunk.is_stale == False)  # noqa: E712
            .order_by(AnalystChunk.created_at.desc())
            .limit(limit)
        )
        chunk_ids = [row[0] for row in result.all()]

        if not chunk_ids:
            logger.info("No chunks to process.")
            return

        logger.info("Processing %d chunks...", len(chunk_ids))
        count = await distill_from_chunks(chunk_ids, db)
        await db.commit()
        logger.info("Done. %d principles created/updated.", count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run principle distillation")
    parser.add_argument("--limit", type=int, default=100, help="Max chunks to process")
    args = parser.parse_args()

    asyncio.run(main(args.limit))
