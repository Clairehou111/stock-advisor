"""
Auth API — login with shared password, receive JWT in httpOnly cookie.

V1 auth model:
- Single shared password (from settings)
- On login, find-or-create user by username
- JWT set as httpOnly cookie (not exposed in URL or JS)
- Also returned in response body for backwards compatibility
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.security import create_access_token, get_current_user, hash_password
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


class AuthStatusResponse(BaseModel):
    user_id: str
    username: str | None = None
    is_admin: bool = False


@router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    """Authenticate with shared password. Creates user if username is new."""
    if request.password != settings.shared_password:
        raise HTTPException(status_code=401, detail="Invalid password")

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

    # Set httpOnly cookie — not accessible to JS, not in URL
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        secure=not settings.debug,  # secure=True in production (HTTPS)
        max_age=7 * 24 * 3600,  # 7 days
    )

    return LoginResponse(
        access_token=token,
        user_id=str(user.id),
        is_admin=user.is_admin,
    )


@router.post("/logout")
async def logout(response: Response):
    """Clear the auth cookie."""
    response.delete_cookie("access_token")
    return {"status": "ok"}


@router.get("/me", response_model=AuthStatusResponse)
async def me(current_user: User = Depends(get_current_user)):
    return AuthStatusResponse(
        user_id=str(current_user.id),
        username=current_user.username,
        is_admin=current_user.is_admin,
    )
