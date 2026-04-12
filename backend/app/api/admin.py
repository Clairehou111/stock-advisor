"""
Admin API — protected endpoints for triggering ingestion pipelines.
Requires is_admin=True in JWT.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_current_user
from app.db.session import async_session
from app.models.tables import IngestTask, UploadSource, User

router = APIRouter(prefix="/api/admin", tags=["admin"])
logger = logging.getLogger(__name__)


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


async def _update_task(task_id: str, **kwargs) -> None:
    """Persist task state to DB. kwargs: status, messages (appended), result, error."""
    async with async_session() as db:
        task = await db.get(IngestTask, task_id)
        if not task:
            return
        if "status" in kwargs:
            task.status = kwargs["status"]
        if "message" in kwargs:
            task.messages = list(task.messages or []) + [kwargs["message"]]
        if "result" in kwargs:
            task.result = kwargs["result"]
        if "error" in kwargs:
            task.error = kwargs["error"]
        await db.commit()


# ── Patreon background runner ─────────────────────────────────────────────────

async def _run_patreon_ingest(task_id: str, post_id: str, force: bool = False) -> None:
    from app.ingestion.patreon_parser import ingest_patreon_post
    from app.jobs.distill_principles import distill_from_chunks
    from app.models.tables import AnalystChunk

    async def progress(msg: str) -> None:
        await _update_task(task_id, message=msg)

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

        await _update_task(
            task_id,
            status="done",
            result=result,
            message=(
                f"Complete — {result['chunk_count']} chunks, "
                f"{result['image_count']} charts"
            ),
        )

    except Exception as e:
        logger.exception("Patreon ingestion failed for post %s", post_id)
        await _update_task(task_id, status="error", error=str(e), message=f"Error: {e}")


# ── Patreon endpoints ─────────────────────────────────────────────────────────

class PatreonIngestRequest(BaseModel):
    post_url_or_id: str
    force: bool = False


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
    async with async_session() as db:
        db.add(IngestTask(id=task_id, task_type="patreon", status="running", messages=[], result=None, error=None))
        await db.commit()

    asyncio.create_task(_run_patreon_ingest(task_id, post_id, force=request.force))
    return PatreonTaskResponse(task_id=task_id)


@router.get("/ingest/active")
async def get_active_tasks(
    _admin: User = Depends(_require_admin),
):
    """Return all running or recently finished tasks (last 24h). Called on page load."""
    from datetime import datetime, timedelta, timezone
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    async with async_session() as db:
        result = await db.execute(
            select(IngestTask).where(
                (IngestTask.status == "running") | (IngestTask.created_at >= cutoff)
            ).order_by(IngestTask.created_at.desc())
        )
        tasks = result.scalars().all()
    return [
        {
            "task_id": t.id,
            "task_type": t.task_type,
            "status": t.status,
        }
        for t in tasks
    ]


@router.get("/ingest/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    _admin: User = Depends(_require_admin),
):
    """Poll ingestion task status and progress messages."""
    async with async_session() as db:
        task = await db.get(IngestTask, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskStatusResponse(
        status=task.status,
        messages=task.messages or [],
        result=task.result,
        error=task.error,
    )


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


def _file_sha256(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


async def _find_existing_excel_r2_key(file_sha256: str) -> str | None:
    async with async_session() as db:
        result = await db.execute(
            select(UploadSource.r2_key)
            .where(
                UploadSource.file_type == "xlsx",
                UploadSource.r2_key.is_not(None),
                UploadSource.extracted_json["file_sha256"].astext == file_sha256,
            )
            .order_by(UploadSource.upload_timestamp.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()


# ── Excel background runner ───────────────────────────────────────────────────

async def _run_excel_ingest(task_id: str, file_bytes: bytes, filename: str) -> None:
    from scripts.ingest_excel import ingest as run_excel_ingest

    async def progress(msg: str) -> None:
        await _update_task(task_id, message=msg)

    try:
        file_sha256 = _file_sha256(file_bytes)
        file_size = len(file_bytes)
        safe_name = Path(filename).name

        r2_key = await _find_existing_excel_r2_key(file_sha256)
        if r2_key:
            await progress(f"Duplicate workbook detected (sha256={file_sha256[:12]}...) — reusing existing R2 object: {r2_key}")
        else:
            r2_key = f"excel/{file_sha256}/{safe_name}"
            await progress(f"Uploading to R2: {r2_key}")
            await asyncio.to_thread(_upload_bytes_to_r2, file_bytes, r2_key)

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            async with async_session() as db:
                result = await run_excel_ingest(
                    tmp_path, return_result=True, db=db,
                    progress_cb=progress, r2_key=r2_key,
                    file_sha256=file_sha256, file_size=file_size, original_filename=safe_name,
                )
                await db.commit()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        await _update_task(
            task_id,
            status="done",
            result=result,
            message=f"Complete — {result['stock_count']} stocks, {result['principle_count']} principles.",
        )

    except Exception as e:
        logger.exception("Excel ingestion failed for %s", filename)
        await _update_task(task_id, status="error", error=str(e), message=f"Error: {e}")


# ── Document background runner ────────────────────────────────────────────────

async def _run_doc_ingest(task_id: str, file_bytes: bytes, fname: str) -> None:
    from app.ingestion.doc_parser import ingest_document

    suffix = Path(fname).suffix.lower()

    async def progress(msg: str) -> None:
        await _update_task(task_id, message=msg)

    try:
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
        await _update_task(
            task_id,
            status="done",
            result=result,
            message=f"Complete — {len(chunks)} chunks stored.",
        )

    except Exception as e:
        logger.exception("Document ingestion failed for %s", fname)
        await _update_task(task_id, status="error", error=str(e), message=f"Error: {e}")


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
    async with async_session() as db:
        db.add(IngestTask(id=task_id, task_type="excel", status="running", messages=[], result=None, error=None))
        await db.commit()

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
    async with async_session() as db:
        db.add(IngestTask(id=task_id, task_type="doc", status="running", messages=[], result=None, error=None))
        await db.commit()

    asyncio.create_task(_run_doc_ingest(task_id, file_bytes, fname))
    return PatreonTaskResponse(task_id=task_id)
