"""
Per-user daily rate limiter.

- Each user has a configurable daily_token_limit
- At 80% usage: degrade Gemini Pro → DeepSeek V3
- Resets daily (new date = new row, no cron needed)
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tables import RateLimitUsage, User


class RateLimiter:
    DEGRADATION_THRESHOLD = 0.8  # 80%

    def __init__(self, session: AsyncSession):
        self._session = session

    async def check(self, user_id: uuid.UUID) -> tuple[bool, bool]:
        """
        Check rate limit status for a user.

        Returns:
            (is_allowed, is_degraded)
            - is_allowed: True if user hasn't exceeded daily limit
            - is_degraded: True if user is over 80% and should use cheaper model
        """
        user = await self._session.get(User, user_id)
        if user is None:
            return False, False

        today = date.today()
        usage = await self._get_usage(user_id, today)

        if usage is None:
            return True, False

        ratio = usage.tokens_used / user.daily_token_limit if user.daily_token_limit > 0 else 1.0

        if ratio >= 1.0:
            return False, True  # over limit
        if ratio >= self.DEGRADATION_THRESHOLD:
            return True, True  # allowed but degraded

        return True, False

    async def record_usage(self, user_id: uuid.UUID, tokens: int) -> None:
        """Record token usage for today."""
        today = date.today()

        stmt = pg_insert(RateLimitUsage).values(
            user_id=user_id,
            usage_date=today,
            tokens_used=tokens,
            queries_count=1,
        ).on_conflict_do_update(
            constraint="uq_user_date",
            set_={
                "tokens_used": RateLimitUsage.tokens_used + tokens,
                "queries_count": RateLimitUsage.queries_count + 1,
            },
        )
        await self._session.execute(stmt)
        await self._session.commit()

    async def get_status(self, user_id: uuid.UUID) -> dict:
        """Get current usage status for a user."""
        user = await self._session.get(User, user_id)
        if user is None:
            return {"error": "user not found"}

        today = date.today()
        usage = await self._get_usage(user_id, today)

        tokens_used = usage.tokens_used if usage else 0
        queries = usage.queries_count if usage else 0
        limit = user.daily_token_limit

        return {
            "tokens_used": tokens_used,
            "tokens_limit": limit,
            "queries_today": queries,
            "usage_pct": tokens_used / limit if limit > 0 else 0,
            "is_degraded": tokens_used / limit >= self.DEGRADATION_THRESHOLD if limit > 0 else False,
        }

    async def _get_usage(self, user_id: uuid.UUID, today: date) -> RateLimitUsage | None:
        result = await self._session.execute(
            select(RateLimitUsage).where(
                RateLimitUsage.user_id == user_id,
                RateLimitUsage.usage_date == today,
            )
        )
        return result.scalar_one_or_none()
