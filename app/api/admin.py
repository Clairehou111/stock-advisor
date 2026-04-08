"""
Admin API — protected endpoints for triggering ingestion pipelines.
Requires is_admin=True in JWT.
"""

from __future__ import annotations

import asyncio
import logging
import re
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_current_user
from app.db.session import async_session
from app.models.tables import User

router = APIRouter(prefix="/api/admin", tags=["admin"])
logger = logging.getLogger(__name__)

# ── In-memory task store ──────────────────────────────────────────────────────
# { task_id: { status, messages, result, error } }
_tasks: dict[str, dict] = {}


def _require_admin(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required.")
    return current_user


def _extract_post_id(post_url_or_id: str) -> str:
    match = re.search(r"-(\d+)$", post_url_or_id.rstrip("/"))
    if match:
        return match.group(1)
    if re.match(r"^\d+$", post_url_or_id.strip()):
        return post_url_or_id.strip()
    raise ValueError(f"Cannot extract post ID from: {post_url_or_id!r}")


# ── Patreon background runner ─────────────────────────────────────────────────

async def _run_patreon_ingest(task_id: str, post_id: str, force: bool = False) -> None:
    from app.ingestion.patreon_parser import ingest_patreon_post
    from app.jobs.distill_principles import distill_from_chunks
    from sqlalchemy import select
    from app.models.tables import AnalystChunk

    task = _tasks[task_id]

    async def progress(msg: str) -> None:
        task["messages"].append(msg)

    try:
        async with async_session() as db:
            result = await ingest_patreon_post(post_id, db, progress_cb=progress, force=force)
            await db.flush()

            await progress("Distilling principles...")
            chunks = await db.execute(
                select(AnalystChunk.id).where(
                    AnalystChunk.upload_source_id == result["upload_source_id"]
                )
            )
            chunk_ids = [row[0] for row in chunks.all()]
            if chunk_ids:
                await distill_from_chunks(chunk_ids, db)

            await db.commit()

        task["status"] = "done"
        task["result"] = result
        await progress(
            f"Complete — {result['chunk_count']} chunks, "
            f"{result['signal_count']} signals, "
            f"{result['image_count']} charts"
        )

    except Exception as e:
        logger.exception("Patreon ingestion failed for post %s", post_id)
        task["status"] = "error"
        task["error"] = str(e)
        task["messages"].append(f"Error: {e}")


# ── Patreon endpoints ─────────────────────────────────────────────────────────

class PatreonIngestRequest(BaseModel):
    post_url_or_id: str
    force: bool = False  # True = wipe existing data and re-ingest from scratch


class PatreonTaskResponse(BaseModel):
    task_id: str


class TaskStatusResponse(BaseModel):
    status: str          # "running" | "done" | "error"
    messages: list[str]
    result: dict | None = None
    error: str | None = None


@router.post("/ingest/patreon", response_model=PatreonTaskResponse)
async def ingest_patreon(
    request: PatreonIngestRequest,
    _admin: User = Depends(_require_admin),
):
    """Start Patreon post ingestion as a background task. Returns task_id to poll."""
    try:
        post_id = _extract_post_id(request.post_url_or_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    task_id = str(uuid.uuid4())
    _tasks[task_id] = {"status": "running", "messages": [], "result": None, "error": None}
    asyncio.create_task(_run_patreon_ingest(task_id, post_id, force=request.force))

    return PatreonTaskResponse(task_id=task_id)


@router.get("/ingest/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    _admin: User = Depends(_require_admin),
):
    """Poll ingestion task status and progress messages."""
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskStatusResponse(**task)


# ── R2 upload helper ──────────────────────────────────────────────────────────

def _upload_bytes_to_r2(file_bytes: bytes, r2_key: str) -> None:
    """Upload raw bytes to Cloudflare R2. No-op if R2 is not configured."""
    from app.config import settings
    if not all([settings.r2_endpoint_url, settings.r2_access_key_id,
                settings.r2_secret_access_key, settings.r2_bucket_name]):
        logger.warning("R2 not configured — skipping file backup")
        return
    import boto3
    from botocore.config import Config
    s3 = boto3.client(
        "s3",
        endpoint_url=settings.r2_endpoint_url,
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
        config=Config(signature_version="s3v4"),
    )
    s3.put_object(Bucket=settings.r2_bucket_name, Key=r2_key, Body=file_bytes)
    logger.info("Uploaded %d bytes to R2: %s", len(file_bytes), r2_key)


# ── Excel background runner ───────────────────────────────────────────────────

async def _run_excel_ingest(task_id: str, file_bytes: bytes, filename: str) -> None:
    from scripts.ingest_excel import ingest as run_excel_ingest

    task = _tasks[task_id]

    async def progress(msg: str) -> None:
        task["messages"].append(msg)

    try:
        # Save to R2 first
        r2_key = f"excel/{task_id}/{filename}"
        await progress(f"Uploading to R2: {r2_key}")
        await asyncio.to_thread(_upload_bytes_to_r2, file_bytes, r2_key)

        # Write to temp file for ingest
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            async with async_session() as db:
                result = await run_excel_ingest(
                    tmp_path, return_result=True, db=db,
                    progress_cb=progress, r2_key=r2_key,
                )
                await db.commit()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        task["status"] = "done"
        task["result"] = result
        await progress(
            f"Complete — {result['stock_count']} stocks, "
            f"{result['principle_count']} principles."
        )

    except Exception as e:
        logger.exception("Excel ingestion failed for %s", filename)
        task["status"] = "error"
        task["error"] = str(e)
        task["messages"].append(f"Error: {e}")


# ── Document background runner ────────────────────────────────────────────────

async def _run_doc_ingest(task_id: str, file_bytes: bytes, fname: str) -> None:
    from app.ingestion.doc_parser import ingest_document

    task = _tasks[task_id]
    suffix = Path(fname).suffix.lower()

    async def progress(msg: str) -> None:
        task["messages"].append(msg)

    try:
        # Save to R2 first
        r2_key = f"docs/{task_id}/{fname}"
        await progress(f"Uploading to R2: {r2_key}")
        await asyncio.to_thread(_upload_bytes_to_r2, file_bytes, r2_key)

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            async with async_session() as db:
                chunks = await ingest_document(
                    tmp_path, db,
                    progress_cb=progress, r2_key=r2_key,
                )
                await db.commit()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        upload_source_id = str(chunks[0].upload_source_id) if chunks else ""
        result = {
            "chunk_count": len(chunks),
            "upload_source_id": upload_source_id,
            "file_name": fname,
        }
        task["status"] = "done"
        task["result"] = result
        await progress(f"Complete — {len(chunks)} chunks stored.")

    except Exception as e:
        logger.exception("Document ingestion failed for %s", fname)
        task["status"] = "error"
        task["error"] = str(e)
        task["messages"].append(f"Error: {e}")


# ── Excel endpoint ────────────────────────────────────────────────────────────

@router.post("/ingest/excel", response_model=PatreonTaskResponse)
async def ingest_excel(
    file: UploadFile = File(...),
    _admin: User = Depends(_require_admin),
):
    """Start Excel ingestion as a background task. Returns task_id to poll."""
    if not file.filename or not file.filename.endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Only .xlsx files are supported.")

    file_bytes = await file.read()
    task_id = str(uuid.uuid4())
    _tasks[task_id] = {"status": "running", "messages": [], "result": None, "error": None}
    asyncio.create_task(_run_excel_ingest(task_id, file_bytes, file.filename))
    return PatreonTaskResponse(task_id=task_id)


# ── Document endpoint ─────────────────────────────────────────────────────────

@router.post("/ingest/doc", response_model=PatreonTaskResponse)
async def ingest_doc(
    file: UploadFile = File(...),
    _admin: User = Depends(_require_admin),
):
    """Start document ingestion as a background task. Returns task_id to poll."""
    fname = file.filename or "upload"
    suffix = Path(fname).suffix.lower()
    if suffix not in (".pdf", ".txt", ".md"):
        raise HTTPException(status_code=400, detail="Supported formats: .pdf, .txt, .md")

    file_bytes = await file.read()
    task_id = str(uuid.uuid4())
    _tasks[task_id] = {"status": "running", "messages": [], "result": None, "error": None}
    asyncio.create_task(_run_doc_ingest(task_id, file_bytes, fname))
    return PatreonTaskResponse(task_id=task_id)
