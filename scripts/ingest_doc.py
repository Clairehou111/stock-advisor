"""
CLI: Ingest a PDF/text document into analyst_chunks.

Usage:
    python -m scripts.ingest_doc <path-to-file> [--date YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date

from app.db.session import async_session
from app.ingestion.doc_parser import ingest_document
from app.jobs.distill_principles import distill_from_chunks

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def main(file_path: str, publish_date: date | None) -> None:
    async with async_session() as db:
        chunks = await ingest_document(file_path, db, publish_date=publish_date)
        await db.commit()

        if chunks:
            logger.info("Triggering principle distillation for %d chunks...", len(chunks))
            chunk_ids = [c.id for c in chunks]
            await distill_from_chunks(chunk_ids, db)
            await db.commit()

    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest analyst document into DB")
    parser.add_argument("file_path", help="Path to PDF or text file")
    parser.add_argument("--date", help="Publish date (YYYY-MM-DD)", default=None)
    args = parser.parse_args()

    pub_date = date.fromisoformat(args.date) if args.date else None
    asyncio.run(main(args.file_path, pub_date))
