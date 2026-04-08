"""
Auth API — login with shared password, receive JWT.

V1 auth model:
- Single shared password (from settings)
- On login, find-or-create user by username
- Return JWT for subsequent API calls
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.security import create_access_token, hash_password
from app.db.session import get_db
from app.models.tables import User

router = APIRouter(prefix="/api/auth", tags=["auth"])
logger = logging.getLogger(__name__)


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    is_admin: bool = False


@router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """Authenticate with shared password. Creates user if username is new."""
    # V1: validate against shared password
    if request.password != settings.shared_password:
        raise HTTPException(status_code=401, detail="Invalid password")

    # Find or create user
    result = await db.execute(
        select(User).where(User.username == request.username)
    )
    user = result.scalar_one_or_none()

    if not user:
        user = User(
            username=request.username,
            password_hash=hash_password(request.password),
            daily_token_limit=settings.default_daily_token_limit,
        )
        db.add(user)
        await db.flush()
        logger.info("Created new user: %s", request.username)

    token = create_access_token(user.id, is_admin=user.is_admin)
    await db.commit()

    return LoginResponse(
        access_token=token,
        user_id=str(user.id),
        is_admin=user.is_admin,
    )
