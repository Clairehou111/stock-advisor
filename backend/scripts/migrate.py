"""
One-time database migrations.

Run manually when deploying schema changes:
    python -m scripts.migrate

This keeps app startup clean and non-destructive.
Each migration is idempotent (safe to re-run).
"""

import asyncio
import logging
import sys

from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


MIGRATIONS = [
    # (description, SQL, destructive?)
    (
        "Enable pgvector extension",
        "CREATE EXTENSION IF NOT EXISTS vector",
        False,
    ),
    (
        "Drop legacy price_cache table (replaced by in-memory cache)",
        "DROP TABLE IF EXISTS price_cache",
        True,
    ),
    (
        "Add task_type column to ingest_tasks",
        "ALTER TABLE ingest_tasks ADD COLUMN IF NOT EXISTS task_type VARCHAR(20) DEFAULT 'patreon'",
        False,
    ),
    (
        "Widen trend_status from VARCHAR(255) to TEXT",
        "DO $$ BEGIN "
        "ALTER TABLE stock_predictions ALTER COLUMN trend_status TYPE TEXT; "
        "EXCEPTION WHEN undefined_table THEN NULL; END $$",
        False,
    ),
    (
        "Migrate embedding column to 1024 dimensions (WIPES existing embeddings — re-ingest after)",
        "DO $$ BEGIN "
        "ALTER TABLE analyst_chunks ALTER COLUMN embedding TYPE vector(1024) USING NULL; "
        "EXCEPTION WHEN undefined_table THEN NULL; END $$",
        True,
    ),
    (
        "Create composite index for ticker chunk retrieval",
        "CREATE INDEX IF NOT EXISTS ix_analyst_chunks_ticker_stale_date "
        "ON analyst_chunks (ticker, is_stale, publish_date)",
        False,
    ),
    (
        "Create partial index for philosophy chunk retrieval",
        "CREATE INDEX IF NOT EXISTS ix_analyst_chunks_null_ticker_stale "
        "ON analyst_chunks (is_stale) WHERE ticker IS NULL",
        False,
    ),
    (
        "Create GIN index for tickers_mentioned array overlap",
        "CREATE INDEX IF NOT EXISTS ix_analyst_chunks_tickers_mentioned_gin "
        "ON analyst_chunks USING GIN (tickers_mentioned)",
        False,
    ),
    (
        "Add stock_type column to stock_predictions",
        "ALTER TABLE stock_predictions ADD COLUMN IF NOT EXISTS stock_type VARCHAR(50)",
        False,
    ),
]


async def run_migrations(destructive: bool = False):
    """Run all migrations. Skip destructive ones unless --destructive flag is passed."""
    from app.db.session import engine
    from app.models.base import Base
    import app.models.tables  # noqa: F401

    async with engine.begin() as conn:
        for desc, sql, is_destructive in MIGRATIONS:
            if is_destructive and not destructive:
                logger.info("SKIP (destructive): %s", desc)
                continue
            try:
                logger.info("Running: %s", desc)
                await conn.execute(text(sql))
                logger.info("  OK")
            except Exception as e:
                logger.warning("  FAILED: %s", e)

        # Always run create_all for new tables
        logger.info("Running: Base.metadata.create_all")
        await conn.run_sync(Base.metadata.create_all)
        logger.info("  OK")

    await engine.dispose()


def main():
    destructive = "--destructive" in sys.argv
    if destructive:
        logger.warning("Running with --destructive flag: destructive migrations will execute")
    else:
        logger.info("Running safe migrations only. Use --destructive for schema-breaking changes.")
    asyncio.run(run_migrations(destructive=destructive))
    logger.info("Done.")


if __name__ == "__main__":
    main()
