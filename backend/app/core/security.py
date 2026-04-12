"""
Security utilities — JWT creation/validation, password hashing.

V1 auth: shared password → JWT. Users authenticate with the shared password
and receive a JWT for subsequent requests.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.session import get_db
from app.models.tables import AnonymizationRule, EntityAlias, User

logger = logging.getLogger(__name__)

# ── Seed aliases ──────────────────────────────────────────────────────────────
# Canonical set of known aliases. LLM discoveries are appended to this table at
# runtime — this is just the starting point so common terms work without LLM.

_SEED_ALIASES: list[tuple[str, str, str]] = [
    # (alias, type, value)
    # S&P 500
    ("sp",         "index", "^GSPC"), ("s&p",       "index", "^GSPC"),
    ("s&p 500",    "index", "^GSPC"), ("s&p500",    "index", "^GSPC"),
    ("sp500",      "index", "^GSPC"), ("spx",       "index", "^GSPC"),
    ("spy",        "index", "SPY"),
    # VIX
    ("vix",           "index", "^VIX"), ("fear index",      "index", "^VIX"),
    ("volatility",    "index", "^VIX"), ("volatility index","index", "^VIX"),
    # Dow
    ("dow",        "index", "^DJI"), ("djia",      "index", "^DJI"),
    ("dow jones",  "index", "^DJI"),
    # Nasdaq
    ("nasdaq",     "index", "^IXIC"), ("qqq",      "index", "QQQ"),
    ("ndx",        "index", "^NDX"),  ("nasdaq 100","index", "^NDX"),
    # Russell
    ("russell",       "index", "^RUT"), ("russell 2000", "index", "^RUT"),
    ("iwm",           "index", "IWM"),
    # Rates / commodities
    ("10 year",    "index", "^TNX"), ("10-year",   "index", "^TNX"),
    ("treasury",   "index", "^TNX"),
    ("gold",       "index", "GC=F"), ("xau",       "index", "GC=F"),
    ("oil",        "index", "CL=F"), ("crude",     "index", "CL=F"),
    ("wti",        "index", "CL=F"),
    ("bitcoin",    "index", "BTC-USD"), ("btc",    "index", "BTC-USD"),
    # Common stock nicknames / typos
    ("apple",       "ticker", "AAPL"), ("iphone",       "ticker", "AAPL"),
    ("fruit stock", "ticker", "AAPL"), ("fruit company","ticker", "AAPL"),
    ("nvidia",      "ticker", "NVDA"), ("nvdia",        "ticker", "NVDA"),
    ("gpu stock",   "ticker", "NVDA"), ("jensen",       "ticker", "NVDA"),
    ("microsoft",   "ticker", "MSFT"), ("windows",      "ticker", "MSFT"),
    ("facebook",    "ticker", "META"), ("instagram",    "ticker", "META"),
    ("amazon",      "ticker", "AMZN"), ("aws",          "ticker", "AMZN"),
    ("google",      "ticker", "GOOGL"), ("alphabet",    "ticker", "GOOGL"),
    ("youtube",     "ticker", "GOOGL"),
    ("tesla",       "ticker", "TSLA"), ("elon",         "ticker", "TSLA"),
    ("netflix",     "ticker", "NFLX"),
    ("disney",      "ticker", "DIS"),  ("the mouse",    "ticker", "DIS"),
    ("ibm",         "ticker", "IBM"),  ("big blue",     "ticker", "IBM"),
    ("intel",       "ticker", "INTC"),
    ("amd",         "ticker", "AMD"),  ("advanced micro","ticker","AMD"),
    ("qualcomm",    "ticker", "QCOM"),
    ("broadcom",    "ticker", "AVGO"),
    ("tsmc",        "ticker", "TSM"),  ("taiwan semiconductor","ticker","TSM"),
    ("palantir",    "ticker", "PLTR"),
    ("salesforce",  "ticker", "CRM"),
    ("oracle",      "ticker", "ORCL"),
    ("shopify",     "ticker", "SHOP"),
    ("snowflake",   "ticker", "SNOW"),
    ("crowdstrike", "ticker", "CRWD"),
    ("coinbase",    "ticker", "COIN"),
    ("uber",        "ticker", "UBER"),
    ("airbnb",      "ticker", "ABNB"),
    ("jp morgan",   "ticker", "JPM"),  ("jpmorgan",     "ticker", "JPM"),
    ("goldman",     "ticker", "GS"),   ("goldman sachs","ticker", "GS"),
    ("berkshire",   "ticker", "BRK-B"),("buffett",      "ticker", "BRK-B"),
    ("exxon",       "ticker", "XOM"),  ("exxonmobil",   "ticker", "XOM"),
    ("chevron",     "ticker", "CVX"),
    ("pfizer",      "ticker", "PFE"),
    ("eli lilly",   "ticker", "LLY"),  ("lilly",        "ticker", "LLY"),
]

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

bearer_scheme = HTTPBearer()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def create_access_token(user_id: uuid.UUID, is_admin: bool = False) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    payload = {
        "sub": str(user_id),
        "is_admin": is_admin,
        "exp": expire,
    }
    return jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(HTTPBearer(auto_error=False)),
    db: AsyncSession = Depends(get_db),
) -> User:
    """FastAPI dependency: extract and validate user from JWT.

    Checks (in order):
    1. httpOnly cookie 'access_token'
    2. Authorization: Bearer <token> header
    """
    token = None

    # 1. Try cookie first
    cookie_token = request.cookies.get("access_token")
    if cookie_token:
        token = cookie_token

    # 2. Fall back to Authorization header
    if not token and credentials:
        token = credentials.credentials

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    payload = decode_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    user = await db.get(User, uuid.UUID(user_id))
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user


async def ensure_aliases_seeded(db: AsyncSession) -> None:
    """Insert seed aliases that don't already exist. Idempotent."""
    existing = await db.execute(select(EntityAlias.alias))
    existing_set = {row[0] for row in existing.all()}
    new_entries = [
        EntityAlias(alias=alias, resolved_type=rtype, resolved_value=value)
        for alias, rtype, value in _SEED_ALIASES
        if alias not in existing_set
    ]
    if new_entries:
        db.add_all(new_entries)
        await db.commit()


async def ensure_admin_exists(db: AsyncSession) -> None:
    """Ensure the configured admin_username has is_admin=True. Called on startup.

    - If the user exists → promote to admin (idempotent).
    - If not → create with shared_password and is_admin=True.
    """
    admin_username = settings.admin_username

    # Check if this user already exists
    result = await db.execute(select(User).where(User.username == admin_username))
    user = result.scalar_one_or_none()

    if user:
        if not user.is_admin:
            user.is_admin = True
            await db.commit()
        return

    # Create new admin user
    admin = User(
        username=admin_username,
        password_hash=hash_password(settings.shared_password),
        is_admin=True,
        daily_token_limit=settings.default_daily_token_limit,
    )
    db.add(admin)
    await db.commit()


async def ensure_anon_rules_seeded(db: AsyncSession) -> None:
    """Seed AnonymizationRule table from ANON_EXTRA_RULES env var. Idempotent.

    ANON_EXTRA_RULES should be a JSON array of [term, replacement, category] tuples.
    Use plain text terms — word boundaries are added automatically for name/handle categories.
    Example: [["Real Name", "Sid Sloth", "name"], ["real_handle", "sid_sloth", "handle"]]
    Set this in .env or Railway environment variables — never commit real names to code.
    """
    import re as _re

    if not settings.anon_extra_rules:
        return

    try:
        raw_rules = json.loads(settings.anon_extra_rules)
    except Exception:
        logger.warning("ANON_EXTRA_RULES is not valid JSON — skipping anon rule seeding")
        return

    existing = await db.execute(select(AnonymizationRule.original_term))
    existing_set = {row[0] for row in existing.all()}

    new_entries = []
    for rule in raw_rules:
        if len(rule) < 2:
            continue
        term, replacement = rule[0], rule[1]
        category = rule[2] if len(rule) > 2 else "name"
        # Build regex: word-boundary wrap for name/handle, literal pattern otherwise
        if category in ("name", "handle"):
            pattern = r"\b" + _re.escape(term) + r"\b"
        else:
            pattern = term
        if pattern not in existing_set:
            new_entries.append(AnonymizationRule(
                original_term=pattern,
                replacement=replacement,
                category=category,
            ))
    if new_entries:
        db.add_all(new_entries)
        await db.commit()
        logger.info("Seeded %d anonymization rules from env var", len(new_entries))


async def load_anon_rules_into_memory(db: AsyncSession) -> None:
    """Load all AnonymizationRule records from DB into the in-memory Anonymizer.
    Call this at startup after ensure_anon_rules_seeded.
    """
    from app.ingestion.anonymizer import set_runtime_rules

    result = await db.execute(select(AnonymizationRule))
    rules = [
        (r.original_term, r.replacement, r.category)
        for r in result.scalars().all()
    ]
    set_runtime_rules(rules)
    if rules:
        logger.info("Loaded %d anonymization rules into memory", len(rules))
