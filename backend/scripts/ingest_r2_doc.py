"""
CLI: Download a document from Cloudflare R2 and ingest it into analyst_chunks.

Usage:
    python -m scripts.ingest_r2_doc <r2-key> [--date YYYY-MM-DD]

Example:
    python -m scripts.ingest_r2_doc "patron/doc/Oil War.pdf" --date 2026-04-01
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import tempfile
from datetime import date
from pathlib import Path

import boto3
from botocore.config import Config

from app.config import settings
from app.db.session import async_session
from app.ingestion.doc_parser import ingest_document
from app.jobs.distill_principles import distill_from_chunks

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def download_from_r2(r2_key: str, local_path: Path) -> None:
    """Download a file from Cloudflare R2 to a local path."""
    s3 = boto3.client(
        "s3",
        endpoint_url=settings.r2_endpoint_url,
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )
    logger.info("Downloading s3://%s/%s ...", settings.r2_bucket_name, r2_key)
    s3.download_file(settings.r2_bucket_name, r2_key, str(local_path))
    logger.info("Downloaded to %s (%.1f KB)", local_path, local_path.stat().st_size / 1024)


async def main(r2_key: str, publish_date: date | None) -> None:
    suffix = Path(r2_key).suffix or ".pdf"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        download_from_r2(r2_key, tmp_path)

        async with async_session() as db:
            chunks = await ingest_document(tmp_path, db, publish_date=publish_date)

            # Store r2_key on the upload_source record
            if chunks and chunks[0].upload_source_id:
                from app.models.tables import UploadSource
                from sqlalchemy import select
                result = await db.execute(
                    select(UploadSource).where(UploadSource.id == chunks[0].upload_source_id)
                )
                src = result.scalar_one_or_none()
                if src:
                    src.r2_key = r2_key

            await db.commit()

            if chunks:
                logger.info("Triggering principle distillation for %d chunks...", len(chunks))
                chunk_ids = [c.id for c in chunks]
                await distill_from_chunks(chunk_ids, db)
                await db.commit()
                logger.info("Done. %d chunks ingested, principles updated.", len(chunks))
            else:
                logger.warning("No chunks extracted from document.")
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a document from Cloudflare R2")
    parser.add_argument("r2_key", help="R2 object key (e.g. patron/doc/file.pdf)")
    parser.add_argument("--date", help="Publish date YYYY-MM-DD", default=None)
    args = parser.parse_args()

    pub_date = date.fromisoformat(args.date) if args.date else None
    asyncio.run(main(args.r2_key, pub_date))
