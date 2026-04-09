from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.admin import router as admin_router
from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.config import settings
from app.core.security import (
    ensure_admin_exists,
    ensure_aliases_seeded,
    ensure_anon_rules_seeded,
    load_anon_rules_into_memory,
)
from app.db.session import async_session, engine
from app.models.base import Base
import app.models.tables  # noqa: F401 — ensure all models are registered in Base.metadata


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure pgvector extension + all tables exist
    async with engine.begin() as conn:
        await conn.execute(
            __import__("sqlalchemy").text("CREATE EXTENSION IF NOT EXISTS vector")
        )
        # price_cache replaced by in-memory cache — drop if still exists
        await conn.execute(
            __import__("sqlalchemy").text("DROP TABLE IF EXISTS price_cache")
        )
        # Add task_type column to ingest_tasks if missing
        await conn.execute(
            __import__("sqlalchemy").text(
                "ALTER TABLE ingest_tasks ADD COLUMN IF NOT EXISTS task_type VARCHAR(20) DEFAULT 'patreon'"
            )
        )
        # Migrate embedding column from 768 to 1024 dims (one-time, safe on fresh deploy)
        await conn.execute(
            __import__("sqlalchemy").text(
                "DO $$ BEGIN "
                "ALTER TABLE analyst_chunks ALTER COLUMN embedding TYPE vector(1024) USING NULL; "
                "EXCEPTION WHEN undefined_table THEN NULL; END $$"
            )
        )
        # Widen trend_status from VARCHAR(255) to TEXT (holds LLM-generated market commentary)
        await conn.execute(
            __import__("sqlalchemy").text(
                "DO $$ BEGIN "
                "ALTER TABLE stock_predictions ALTER COLUMN trend_status TYPE TEXT; "
                "EXCEPTION WHEN undefined_table THEN NULL; END $$"
            )
        )
        # Create indexes for 3-channel chunk retrieval (safe if already exist)
        await conn.execute(
            __import__("sqlalchemy").text(
                "CREATE INDEX IF NOT EXISTS ix_analyst_chunks_ticker_stale_date "
                "ON analyst_chunks (ticker, is_stale, publish_date)"
            )
        )
        await conn.execute(
            __import__("sqlalchemy").text(
                "CREATE INDEX IF NOT EXISTS ix_analyst_chunks_null_ticker_stale "
                "ON analyst_chunks (is_stale) WHERE ticker IS NULL"
            )
        )
        await conn.execute(
            __import__("sqlalchemy").text(
                "CREATE INDEX IF NOT EXISTS ix_analyst_chunks_tickers_mentioned_gin "
                "ON analyst_chunks USING GIN (tickers_mentioned)"
            )
        )
        await conn.run_sync(Base.metadata.create_all)
    # Mark any tasks that were "running" when the server last died as errors
    from sqlalchemy import update as sa_update
    from app.models.tables import IngestTask
    async with async_session() as db:
        await db.execute(
            sa_update(IngestTask)
            .where(IngestTask.status == "running")
            .values(status="error", error="Server restarted — task was interrupted.")
        )
        await db.commit()

    # Seed admin user, alias cache, and anonymization rules
    async with async_session() as db:
        await ensure_admin_exists(db)
        await ensure_aliases_seeded(db)
        await ensure_anon_rules_seeded(db)
        await load_anon_rules_into_memory(db)
    yield
    # Shutdown
    await engine.dispose()


app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(admin_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
