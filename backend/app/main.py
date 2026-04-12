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
    """App startup: non-destructive bootstrapping only.

    For schema migrations (ALTER TABLE, CREATE INDEX, etc.),
    run `python -m scripts.migrate` before deploying.
    """
    from sqlalchemy import text, update as sa_update
    from app.models.tables import IngestTask

    # Ensure pgvector extension + create any new tables
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    # Mark interrupted tasks as errors
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
    await engine.dispose()


app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
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
