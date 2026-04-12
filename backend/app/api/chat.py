"""
Chat API — the main endpoint for user queries.

Flow:
1. Detect tickers in query
2. Fetch structured facts from stock_predictions
3. Run decision engine (Stage 1 math)
4. Retrieve relevant analyst chunks via pgvector (RAG)
5. Load principle corpus
6. Build system prompt with Carlin persona
7. Route to appropriate LLM (Gemini/DeepSeek)
8. Anonymization post-check
9. Record usage + save message
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import case, func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.decision_engine import StockData, analyze_ticker
from app.db.session import async_session, get_db
from app.ingestion.anonymizer import Anonymizer
from app.llm.orchestrator import chat as llm_chat, select_model
from app.llm.prompts import build_system_prompt, format_metrics, format_stock_context
from app.models.tables import (
    AnalystChunk,
    Conversation,
    DerivedPrinciple,
    EntityAlias,
    Message,
    PortfolioHolding,
    PrincipleCorpus,
    StockPrediction,
    User,
)
from app.core.security import get_current_user
from app.services.earnings_service import format_earnings_notice, get_upcoming_earnings
from app.services.embedding_service import embed_text
from app.services.price_service import get_price, get_ticker_meta
from app.services.rate_limiter import RateLimiter

router = APIRouter(prefix="/api", tags=["chat"])
logger = logging.getLogger(__name__)

# NOTE: do NOT instantiate Anonymizer() at module level — runtime rules (sensitive
# name patterns) are loaded via set_runtime_rules() during app startup lifespan,
# which runs AFTER module import. Always instantiate inside request/task scope.

# Visual chart references to strip before injecting chunks into the LLM prompt.
# The DB keeps the raw descriptions; users never see chart images so these are meaningless.
_VISUAL_REF_RE = re.compile(
    r"\(\s*(?:horizontal|vertical|diagonal|dashed|dotted|solid)?\s*"
    r"(?:red|green|blue|orange|yellow|purple|black|white|grey|gray)\s*"
    r"(?:horizontal|vertical|diagonal|dashed|dotted|solid)?\s*"
    r"(?:line|arrow|bar|zone|band|area|channel|box|support|resistance)?\s*\)",
    re.I,
)
_VISUAL_WORD_RE = re.compile(
    r"\b(?:horizontal|vertical|diagonal|dashed|dotted)\s+"
    r"(?:red|green|blue|orange|yellow|purple|black|white|grey|gray)\s+"
    r"(?:line|arrow|bar|zone|band|area|channel|box)\b",
    re.I,
)


def _clean_chunk_text(text: str) -> str:
    """Strip chart visual references (colors/line styles) from chunk text."""
    text = _VISUAL_REF_RE.sub("", text)
    text = _VISUAL_WORD_RE.sub("level", text)
    # Collapse any double spaces left behind
    text = re.sub(r"  +", " ", text)
    return text.strip()



# ── Message annotation (rule-based, zero LLM cost) ───────────────────────────

_INTENT_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"covered\s+(?:call|short)|wheel|options|puts?|calls?", re.I), "options_strategy"),
    (re.compile(r"\bshort\b|put|bear", re.I), "bearish_strategy"),
    (re.compile(r"\bbuy\b|accumulate|long\b|dip\b", re.I), "bullish_strategy"),
    (re.compile(r"compare|vs\b|\bor\b.*which", re.I), "comparison"),
    (re.compile(r"price|trading\s+at|worth|quote", re.I), "price_query"),
    (re.compile(r"\bpe\b|valuation|fair\s+value|overvalued|undervalued", re.I), "valuation"),
]

_SENTIMENT_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"crash|dump|sell\s*off|bearish|downside|risk|fear", re.I), "bearish"),
    (re.compile(r"rally|moon|bullish|upside|breakout|accumulate", re.I), "bullish"),
]


def _classify_message_metadata(text: str, tickers: list[str]) -> dict:
    """Rule-based classification of intent + sentiment. No LLM call."""
    intent = "general"
    for pattern, label in _INTENT_RULES:
        if pattern.search(text):
            intent = label
            break

    sentiment = "neutral"
    for pattern, label in _SENTIMENT_RULES:
        if pattern.search(text):
            sentiment = label
            break

    return {"intent": intent, "sentiment": sentiment, "tickers": tickers}


# ── LLM entity extraction ─────────────────────────────────────────────────────
# One small LLM call resolves typos, nicknames, and index shorthands.
# This replaces all static alias dicts — no maintenance needed.

_ENTITY_PROMPT = """\
You are a financial query parser. Extract all stocks and market indices from the user's query.
Correct any typos and resolve nicknames to official symbols.
For each match, return the exact alias text the user wrote (lowercased) alongside the resolved symbol.

For market indices use these symbols:
^GSPC = S&P 500 (sp, spx, s&p, s&p500)
^VIX  = VIX (vix, fear index, volatility index)
^DJI  = Dow Jones (dow, djia)
^IXIC = Nasdaq Composite (nasdaq)
^NDX  = Nasdaq 100 (ndx)
^RUT  = Russell 2000 (russell)
^TNX  = 10-Year Treasury (10 year, treasury)
GC=F  = Gold (gold, xau)
CL=F  = Oil (oil, crude, wti)
BTC-USD = Bitcoin (bitcoin, btc)
SPY = SPY ETF, QQQ = QQQ ETF

For stocks return official ticker. Examples:
"apple" / "fruit stock" → AAPL, "nvdia" → NVDA, "the mouse" → DIS, "big blue" → IBM
{history_context}
Return ONLY this JSON:
{{"found": [{{"alias": "sp", "type": "index", "value": "^GSPC"}}], "needs_history": false, "is_social": false}}

- "needs_history": true ONLY if the query is a follow-up that clearly refers to a previously discussed stock \
(e.g. "can i short it", "what about long term", "how much to buy"). \
false if the query is general/educational (e.g. "what is pe", "explain fair value") or mentions a specific ticker already.
- "is_social": true if the query is purely social/acknowledgment with no real question \
(e.g. "great!", "thanks", "ok cool", "lovely", "got it", "haha"). false otherwise.

Query: "{query}\""""

# Display names for index symbols
_INDEX_DISPLAY: dict[str, str] = {
    "^GSPC": "S&P 500", "^VIX": "VIX", "^DJI": "Dow Jones",
    "^IXIC": "Nasdaq Composite", "^NDX": "Nasdaq 100", "^RUT": "Russell 2000",
    "^TNX": "10-Year Treasury Yield", "GC=F": "Gold Futures",
    "CL=F": "Crude Oil Futures", "BTC-USD": "Bitcoin",
    "SPY": "SPY (S&P 500 ETF)", "QQQ": "QQQ (Nasdaq ETF)", "IWM": "IWM (Russell 2000 ETF)",
}


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    conversation_id: str
    model_used: str
    tokens_used: int
    is_degraded: bool = False


class ConversationSummaryResponse(BaseModel):
    id: str
    title: str | None = None
    created_at: datetime
    last_message_at: datetime | None = None
    message_count: int = 0


class ConversationListResponse(BaseModel):
    conversations: list[ConversationSummaryResponse]


class ConversationMessageResponse(BaseModel):
    id: str
    role: str
    content: str
    model_used: str | None = None
    tokens_used: int | None = None
    tickers_mentioned: list[str] | None = None
    metadata_json: dict | None = None
    created_at: datetime


class ConversationDetailResponse(BaseModel):
    conversation_id: str
    title: str | None = None
    summary: str | None = None
    total_message_count: int = 0
    truncated: bool = False
    messages: list[ConversationMessageResponse]


def _message_role_rank():
    return case(
        (Message.role == "user", 0),
        (Message.role == "assistant", 1),
        else_=2,
    )


def _query_ngrams(query: str) -> list[str]:
    """Extract 1-, 2-, and 3-word lowercase n-grams from query.

    Also extracts embedded ASCII tokens from mixed-script tokens (e.g. "告诉我SP的价格" → "sp").
    This handles Chinese/Japanese/Korean text that contains Latin ticker symbols without spaces.
    """
    cleaned = re.sub(r"[^\w\s&]", " ", query.lower())
    words = cleaned.split()

    # For any token that mixes scripts, also extract pure-ASCII sub-tokens
    extra: list[str] = []
    for w in words:
        if re.search(r"[^\x00-\x7f]", w):  # contains non-ASCII (CJK etc.)
            ascii_parts = re.findall(r"[a-z0-9&][a-z0-9&]*", w)
            extra.extend(ascii_parts)
    words = words + extra

    ngrams = []
    for i, w in enumerate(words):
        ngrams.append(w)
        if i + 1 < len(words):
            ngrams.append(f"{w} {words[i+1]}")
        if i + 2 < len(words):
            ngrams.append(f"{w} {words[i+1]} {words[i+2]}")
    return ngrams


_COMMON_WORDS = {
    # Stop words
    "the", "a", "an", "is", "it", "be", "do", "go", "in", "on", "at", "to",
    "for", "of", "by", "as", "or", "and", "but", "if", "my", "we", "i",
    "me", "us", "he", "she", "are", "was", "has", "had", "did", "got",
    "what", "when", "where", "why", "how", "who", "which", "that", "this",
    "with", "from", "have", "been", "will", "would", "could", "should",
    "today", "now", "just", "up", "down", "about", "any", "all", "get",
    "think", "look", "like", "going", "doing", "than", "then", "also",
    "very", "much", "more", "most", "some", "such", "only", "over",
    "same", "both", "each", "well", "back", "been", "here", "there",
    "yes", "no", "ok", "okay", "sure", "right", "good", "bad", "great",
    "hey", "hi", "hello", "thanks", "thank", "sorry", "please", "can",
    "tell", "show", "give", "help", "know", "want", "need", "let",
    "say", "said", "make", "take", "come", "keep", "still", "into",
    # Financial common words (not tickers)
    "price", "market", "stock", "stocks", "index", "fund", "etf", "crypto",
    "money", "buy", "sell", "hold", "long", "short", "call", "put",
    "bull", "bear", "dip", "drop", "fall", "rise", "gain", "loss",
    "zone", "range", "level", "high", "low", "top", "peak", "trim",
    "cost", "value", "fair", "cheap", "risk", "safe",
    "trend", "move", "open", "close", "trade", "wait", "watch",
    "plunge", "crash", "rally", "surge", "dump", "pump", "moon",
    "current", "recent", "next", "last", "first", "year", "month",
    "week", "day", "time", "ago", "soon", "early", "late",
    "portfolio", "position", "exposure", "allocation",
}


async def _extract_entities(
    query: str, db: AsyncSession,
    prev_tickers: list[str] | None = None,
) -> tuple[list[str], list[tuple[str, str]], bool, bool]:
    """Resolve tickers and indices from query.

    1. DB lookup: check entity_aliases for all n-grams (instant, no LLM).
    2. LLM fallback: only called when unrecognised financial mentions remain.
       New aliases from LLM are saved back to DB for future queries.

    Returns:
        (tickers, indices, needs_history, is_social)
    """
    ngrams = _query_ngrams(query)

    # ── Step 1: batch DB lookup ───────────────────────────────────────────────
    result = await db.execute(
        select(EntityAlias).where(EntityAlias.alias.in_(ngrams))
    )
    db_hits: dict[str, EntityAlias] = {row.alias: row for row in result.scalars().all()}

    tickers: list[str] = []
    index_map: dict[str, str] = {}  # symbol → display name (deduplicated)

    for row in db_hits.values():
        if row.resolved_type == "ticker":
            tickers.append(row.resolved_value)
        else:
            index_map[row.resolved_value] = _INDEX_DISPLAY.get(row.resolved_value, row.resolved_value)

    # ── Step 2: decide if LLM needed ─────────────────────────────────────────
    # Only consider single-word unresolved tokens that look like tickers
    # (1-6 uppercase letters, not common words, not numbers)
    unresolved_ticker_shaped = [
        t for t in ngrams
        if t not in db_hits
        and t not in _COMMON_WORDS
        and len(t.split()) == 1   # single word only
        and len(t) <= 6
        and not t.isdigit()
        and any(c.isalpha() for c in t)
    ]
    # LLM needed only if we have ticker-shaped unknowns AND no tickers found yet
    needs_llm = bool(unresolved_ticker_shaped) and not (tickers or index_map)

    # Follow-up detection: if prev_tickers exist, no tickers found, AND
    # query has financial intent words — likely a follow-up, carry prev_tickers instead of calling LLM
    _FOLLOWUP_WORDS = {"it", "its", "this", "that", "them", "those", "same", "again",
                       "short", "buy", "sell", "trim", "hold", "accumulate", "covered"}
    if not needs_llm and not tickers and not index_map and prev_tickers:
        query_words = set(query.lower().split())
        if query_words & _FOLLOWUP_WORDS:
            # Looks like a follow-up — carry prev_tickers directly, skip LLM
            needs_llm = False  # will be handled by the caller's prev_tickers carry logic
        else:
            needs_llm = True  # genuinely unknown query, ask LLM

    needs_history = False
    is_social = False

    # For detected follow-ups (matched _FOLLOWUP_WORDS), signal needs_history
    # so the caller carries prev_tickers without an LLM call
    if not needs_llm and not tickers and not index_map and prev_tickers:
        query_words = set(query.lower().split())
        if query_words & _FOLLOWUP_WORDS:
            needs_history = True

    if needs_llm and settings.deepseek_api_key:
        import httpx
        history_ctx = ""
        if prev_tickers:
            history_ctx = (
                f"\nThe user was previously discussing: {', '.join(prev_tickers)}. "
                "Consider whether this new query refers to those tickers or is an independent question.\n"
            )
        prompt = _ENTITY_PROMPT.format(query=query, history_context=history_ctx)
        try:
            async with httpx.AsyncClient(timeout=8.0, trust_env=True) as client:
                resp = await client.post(
                    "https://api.deepseek.com/chat/completions",
                    headers={"Authorization": f"Bearer {settings.deepseek_api_key}"},
                    json={
                        "model": "deepseek-chat",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 150,
                        "temperature": 0.0,
                    },
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
                data = json.loads(raw)

            # LLM returns {"found": [...], "needs_history": true/false, "is_social": true/false}
            needs_history = bool(data.get("needs_history", False))
            is_social = bool(data.get("is_social", False))
            new_entries = []
            for item in data.get("found", []):
                alias = item.get("alias", "").lower().strip()
                rtype = item.get("type", "").strip()
                value = item.get("value", "").upper().strip()
                if not alias or not value:
                    continue
                # Normalize type — LLM sometimes returns "stock" instead of "ticker"
                if rtype in ("stock", "ticker", "etf"):
                    rtype = "ticker"
                elif rtype != "index":
                    continue
                if alias not in db_hits:
                    new_entries.append(EntityAlias(alias=alias, resolved_type=rtype, resolved_value=value))
                    db_hits[alias] = new_entries[-1]  # avoid duplicates in this request
                if rtype == "ticker":
                    tickers.append(value)
                else:
                    index_map[value] = _INDEX_DISPLAY.get(value, value)

            if new_entries:
                # ON CONFLICT DO NOTHING — handles race conditions and seeded aliases
                # (e.g. "sp" seeded at startup but not matched via n-gram in Chinese text)
                await db.execute(
                    pg_insert(EntityAlias)
                    .values([
                        {"alias": e.alias, "resolved_type": e.resolved_type, "resolved_value": e.resolved_value}
                        for e in new_entries
                    ])
                    .on_conflict_do_nothing(index_elements=["alias"])
                )
                logger.info("Saved %d new entity aliases: %s", len(new_entries),
                            [e.alias for e in new_entries])

        except Exception as e:
            logger.warning("Entity LLM fallback failed: %s", e)

    # If DB resolved tickers, no need for history (we found what we need)
    return list(set(tickers)), list(index_map.items()), needs_history, is_social


async def _check_coverage(tickers: list[str], db: AsyncSession) -> tuple[list[str], list[str]]:
    """Split resolved tickers into covered (in DB) vs uncovered."""
    if not tickers:
        return [], []
    result = await db.execute(
        select(StockPrediction.ticker).where(StockPrediction.is_current == True)  # noqa: E712
    )
    ticker_set = {row[0].split()[0].upper() for row in result.all()}
    covered = [t for t in tickers if t in ticker_set]
    uncovered = [t for t in tickers if t not in ticker_set]
    return covered, uncovered


async def _get_stock_data(ticker: str, db: AsyncSession) -> StockPrediction | None:
    """Fetch current prediction for a ticker."""
    result = await db.execute(
        select(StockPrediction).where(
            StockPrediction.ticker.ilike(ticker),  # exact case-insensitive, no wildcards
            StockPrediction.is_current == True,  # noqa: E712
        )
    )
    return result.scalar_one_or_none()


async def _get_principles(db: AsyncSession) -> list[str]:
    """Load all principles from the corpus."""
    result = await db.execute(select(PrincipleCorpus.principle_text))
    return [row[0] for row in result.all()]


_CHUNK_TOKEN_BUDGET = 3000  # approximate token budget for all retrieved chunks

# Map canonical index symbols → all aliases that might appear as chunk tickers
_INDEX_TICKER_ALIASES: dict[str, list[str]] = {
    "^GSPC": ["^GSPC", "SPX", "S&P", "S&P500", "SP500"],
    "^IXIC": ["^IXIC", "COMPQ", "NASDAQ"],
    "^DJI":  ["^DJI", "DJIA", "DOW"],
    "^VIX":  ["^VIX", "VIX"],
    "^RUT":  ["^RUT", "RUT", "RUSSELL"],
    "^NDX":  ["^NDX", "NDX", "QQQ"],
}


def _expand_index_aliases(tickers: list[str]) -> list[str]:
    """Expand index symbols to include all known aliases for chunk matching."""
    expanded = set(tickers)
    for t in tickers:
        if t in _INDEX_TICKER_ALIASES:
            expanded.update(_INDEX_TICKER_ALIASES[t])
    return list(expanded)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return len(text) // 4


def _format_chunk_with_meta(chunk: AnalystChunk) -> str:
    """Prepend temporal metadata so the LLM can reason about recency."""
    parts = []
    if chunk.publish_date:
        parts.append(str(chunk.publish_date))
    if chunk.temporal_scope:
        parts.append(chunk.temporal_scope)
    if chunk.thesis_direction:
        parts.append(chunk.thesis_direction)
    prefix = f"[{' | '.join(parts)}] " if parts else ""
    return prefix + chunk.content_text


async def _get_relevant_chunks(
    query: str,
    db: AsyncSession,
    tickers: list[str] | None = None,
    uncovered_tickers: list[str] | None = None,
    limit: int = 10,
) -> list[tuple[AnalystChunk, float]]:
    """Three-channel chunk retrieval:

    1. Ticker-tagged chunks — SQL fetch, no embedding. Filtered for staleness.
    2. Cross-reference chunks — ticker appears in tickers_mentioned but is not the primary ticker.
    3. Philosophy/theme chunks — semantic search using ticker description, not raw user message.

    Returns list of (chunk, similarity_score) tuples, ticker-specific first.
    """
    from datetime import date, timedelta

    all_tickers = _expand_index_aliases(
        list(set((tickers or []) + (uncovered_tickers or [])))
    )
    results: list[tuple[AnalystChunk, float]] = []
    seen_ids: set[str] = set()
    token_budget = _CHUNK_TOKEN_BUDGET

    def _add(chunk: AnalystChunk, score: float) -> bool:
        """Add chunk if not seen and within budget. Returns False when budget exhausted."""
        nonlocal token_budget
        if str(chunk.id) in seen_ids:
            return True
        cost = _estimate_tokens(chunk.content_text)
        if cost > token_budget and results:  # always allow at least one
            return False
        seen_ids.add(str(chunk.id))
        results.append((chunk, score))
        token_budget -= cost
        return True

    # ── Channel 1: Ticker-tagged chunks (SQL, no embedding) ──────────────────
    if all_tickers:
        cutoff_90d = date.today() - timedelta(days=90)
        stmt = (
            select(AnalystChunk)
            .where(
                AnalystChunk.ticker.in_(all_tickers),
                AnalystChunk.is_stale == False,  # noqa: E712
                # Filter expired short-term chunks
                ~(
                    (AnalystChunk.temporal_scope == "short_term")
                    & (AnalystChunk.publish_date < cutoff_90d)
                ) | (AnalystChunk.publish_date.is_(None)),
            )
            .order_by(AnalystChunk.publish_date.desc().nulls_last())
        )
        rows = await db.execute(stmt)
        for chunk in rows.scalars().all():
            if not _add(chunk, 1.0):
                break

    # ── Channel 2: Cross-reference chunks (ticker in tickers_mentioned) ──────
    if all_tickers and token_budget > 0:
        stmt = (
            select(AnalystChunk)
            .where(
                AnalystChunk.tickers_mentioned.overlap(all_tickers),
                ~AnalystChunk.ticker.in_(all_tickers) if all_tickers else True,
                AnalystChunk.is_stale == False,  # noqa: E712
            )
            .order_by(AnalystChunk.publish_date.desc().nulls_last())
            .limit(5)
        )
        rows = await db.execute(stmt)
        for chunk in rows.scalars().all():
            if not _add(chunk, 0.8):
                break

    # ── Channel 3: Philosophy/theme chunks via semantic search ───────────────
    if token_budget > 0:
        # Build semantic query: use ticker descriptions (not raw user message)
        if all_tickers:
            meta_tasks = [get_ticker_meta(t) for t in all_tickers[:3]]
            metas = await asyncio.gather(*meta_tasks, return_exceptions=True)
            desc_parts = []
            for t, m in zip(all_tickers[:3], metas):
                if isinstance(m, Exception):
                    desc_parts.append(t)
                else:
                    parts = [m.long_name]
                    if m.sector:
                        parts.append(m.sector)
                    if m.industry:
                        parts.append(m.industry)
                    desc_parts.append(", ".join(parts))
            semantic_query = "analyst view on " + "; ".join(desc_parts)
        else:
            # No ticker — use raw user message (follow-ups, general questions)
            semantic_query = query

        try:
            query_embedding = await embed_text(semantic_query)
        except Exception:
            logger.warning("Embedding failed for philosophy retrieval")
            query_embedding = None

        if query_embedding is not None:
            distance = AnalystChunk.embedding.cosine_distance(query_embedding)
            stmt = (
                select(AnalystChunk, distance.label("distance"))
                .where(
                    AnalystChunk.embedding.isnot(None),
                    AnalystChunk.is_stale == False,  # noqa: E712
                    AnalystChunk.ticker.is_(None),  # philosophy/general chunks only
                )
                .order_by(distance)
                .limit(5)
            )
            rows = await db.execute(stmt)
            for chunk, dist in rows.all():
                similarity = 1.0 - dist
                if similarity < 0.3:  # threshold: skip irrelevant philosophy
                    continue
                if not _add(chunk, similarity):
                    break

    return results


async def _update_chunk_quality(
    chunks_with_scores: list[tuple[AnalystChunk, float]],
    db: AsyncSession,
) -> None:
    """Update retrieval_count, avg_relevance, last_retrieved for retrieved chunks."""
    now = datetime.now(timezone.utc)
    for chunk, score in chunks_with_scores:
        chunk.retrieval_count = (chunk.retrieval_count or 0) + 1
        # Rolling average: new_avg = old_avg + (score - old_avg) / count
        old_avg = float(chunk.avg_relevance) if chunk.avg_relevance else score
        chunk.avg_relevance = old_avg + (score - old_avg) / chunk.retrieval_count
        chunk.last_retrieved = now


_SUMMARY_PROMPT = """\
Analyze the following conversation between a user and a stock advisor.

Produce a JSON response with two fields:
1. "summary": A 2-5 sentence summary capturing stocks discussed, conclusions, user goals/constraints.
2. "context_map": An object mapping each ticker/index discussed to a 1-sentence summary of what was said about it.

Example output:
{{"summary": "The user explored shorting strategies for SOXS and TSLA...", "context_map": {{"SOXS": "3x leveraged bear ETF, user asked about shorting risk.", "TSLA": "Covered short discussion, PE extended."}}}}

Conversation:
{messages}

Return ONLY the JSON:"""

_SUMMARIZE_AFTER_TURNS = 6   # summarize once we have this many stored turns
_KEEP_RECENT_TURNS = 3       # always keep this many recent turns verbatim


async def _summarize_older_messages(
    conversation_id: uuid.UUID,
    db: AsyncSession,
) -> None:
    """Background task: summarize messages beyond the recent window and update Conversation.summary."""
    import httpx

    conv_result = await db.execute(select(Conversation).where(Conversation.id == conversation_id))
    conv = conv_result.scalar_one_or_none()
    if not conv:
        return

    all_msgs = await db.execute(
        select(Message.role, Message.content)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    rows = all_msgs.all()
    total = len(rows)

    # Only summarize messages older than the recent window
    to_summarize = rows[: total - _KEEP_RECENT_TURNS * 2]
    if len(to_summarize) <= (conv.summarized_through or 0):
        return  # nothing new to summarize

    messages_text = "\n".join(
        f"{r.role.upper()}: {r.content[:500]}" for r in to_summarize
    )
    prompt = _SUMMARY_PROMPT.format(messages=messages_text)

    try:
        async with httpx.AsyncClient(timeout=15.0, trust_env=True) as client:
            resp = await client.post(
                "https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {settings.deepseek_api_key}"},
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 400,
                    "temperature": 0.3,
                },
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        try:
            data = json.loads(raw)
            conv.summary = data.get("summary", raw)
            if data.get("context_map"):
                # Merge new context map with existing (don't lose old tickers)
                existing_map = conv.context_map or {}
                existing_map.update(data["context_map"])
                conv.context_map = existing_map
        except (json.JSONDecodeError, AttributeError):
            # Fallback: treat the whole response as a flat summary
            conv.summary = raw

        conv.summarized_through = len(to_summarize)
        await db.commit()
        logger.info("Summarized conversation %s (%d messages, %d ticker contexts)",
                     conversation_id, len(to_summarize),
                     len(conv.context_map) if conv.context_map else 0)
    except Exception as e:
        logger.warning("Conversation summarization failed: %s", e)


async def _load_conversation_history(
    conversation_id: str,
    db: AsyncSession,
    current_tickers: list[str] | None = None,
) -> list[dict]:
    """Load conversation history with ticker-aware filtering.

    - Most recent 1 turn: always verbatim (conversational continuity)
    - Older turns: keep verbatim if tickers overlap with current_tickers,
      otherwise compress to a one-line marker
    - Per-ticker context map entries injected for current tickers
    - Falls back to include-all when current_tickers is empty (follow-ups)
    """
    conv_uuid = uuid.UUID(conversation_id)

    conv_result = await db.execute(select(Conversation).where(Conversation.id == conv_uuid))
    conv = conv_result.scalar_one_or_none()
    if not conv:
        return []

    # Load recent messages with ticker tags
    recent_result = await db.execute(
        select(Message.role, Message.content, Message.tickers_mentioned)
        .where(Message.conversation_id == conv_uuid)
        .order_by(Message.created_at.desc())
        .limit(_KEEP_RECENT_TURNS * 2)
    )
    rows = list(reversed(recent_result.all()))

    messages: list[dict] = []

    # Inject a list of ALL tickers discussed in this conversation (for meta-queries like "list all stocks")
    all_tickers_result = await db.execute(
        text(
            "SELECT ARRAY(SELECT DISTINCT unnest(tickers_mentioned) "
            "FROM messages WHERE conversation_id = :cid AND tickers_mentioned IS NOT NULL)"
        ),
        {"cid": conv_uuid},
    )
    all_conv_tickers = all_tickers_result.scalar()
    if all_conv_tickers:
        messages.append({
            "role": "system",
            "content": f"## All tickers discussed in this conversation\n{', '.join(sorted(all_conv_tickers))}",
        })

    # Inject per-ticker context map (preferred over flat summary)
    if conv.context_map and current_tickers:
        relevant = {t: conv.context_map[t] for t in current_tickers if t in (conv.context_map or {})}
        if relevant:
            ctx_lines = [f"- {t}: {desc}" for t, desc in relevant.items()]
            messages.append({
                "role": "system",
                "content": "## Context from earlier in this conversation\n" + "\n".join(ctx_lines),
            })
    elif conv.summary:
        messages.append({
            "role": "system",
            "content": f"## Earlier in this conversation\n{conv.summary}",
        })

    # Group into turn pairs and filter by ticker relevance
    current_set = set(t.upper() for t in current_tickers) if current_tickers else None

    for i in range(0, len(rows), 2):
        is_most_recent_pair = (i >= len(rows) - 2)
        user_row = rows[i] if i < len(rows) else None
        asst_row = rows[i + 1] if i + 1 < len(rows) else None

        if not user_row:
            continue

        msg_tickers = set(t.upper() for t in (user_row.tickers_mentioned or [])) if user_row.tickers_mentioned else None

        # Decide: include verbatim or compress?
        include_verbatim = (
            is_most_recent_pair          # always keep last turn for flow
            or current_set is None       # no filtering (follow-up question)
            or msg_tickers is None       # legacy message, no tags — include for safety
            or bool(current_set & msg_tickers)  # ticker overlap
        )

        if include_verbatim:
            messages.append({"role": user_row.role, "content": user_row.content})
            if asst_row:
                messages.append({"role": asst_row.role, "content": asst_row.content})
        else:
            # Compress off-topic turn to a marker
            off_tickers = ", ".join(sorted(msg_tickers)) if msg_tickers else "other topics"
            messages.append({
                "role": "system",
                "content": f"[Earlier: discussed {off_tickers} — details omitted for focus.]",
            })

    return messages


async def _tickers_from_history(
    conversation_id: str, db: AsyncSession
) -> tuple[list[str], list[str]]:
    """Get tickers from the most recent tagged user message in the conversation.

    Uses the tickers_mentioned tags saved at write time — only returns tickers
    the user actually asked about, not tickers the assistant mentioned in passing.
    Falls back to text extraction for legacy untagged messages.
    """
    # Find most recent user message with ticker tags
    result = await db.execute(
        select(Message.tickers_mentioned, Message.content)
        .where(
            Message.conversation_id == uuid.UUID(conversation_id),
            Message.role == "user",
        )
        .order_by(Message.created_at.desc())
        .limit(3)
    )
    rows = result.all()

    # Try tagged messages first
    for row in rows:
        if row.tickers_mentioned:
            return await _check_coverage(row.tickers_mentioned, db)

    # Fallback for legacy untagged messages: extract from user message text only
    for row in rows:
        if row.content:
            all_t, _, _, _ = await _extract_entities(row.content, db)
            if all_t:
                return await _check_coverage(all_t, db)

    return [], []


async def _get_portfolio_context(user_id: uuid.UUID, db: AsyncSession) -> str:
    """Load user's portfolio holdings for context."""
    result = await db.execute(
        select(PortfolioHolding).where(PortfolioHolding.user_id == user_id)
    )
    holdings = result.scalars().all()
    if not holdings:
        return ""

    lines = ["## Your Portfolio"]
    for h in holdings:
        cost = f" (avg cost ${float(h.avg_cost_basis):.2f})" if h.avg_cost_basis else ""
        lines.append(f"- {h.ticker}: {float(h.shares):.1f} shares{cost}")
    return "\n".join(lines)


# ── Post-generation fact check (zero LLM cost) ───────────────────────────────

_PRICE_CLAIM_RE = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)")
_PE_CLAIM_RE = re.compile(r"(?:PE|P/E|pe ratio)\s*(?:of|is|at|:)?\s*([\d.]+)", re.I)
_TICKER_MENTION_RE = re.compile(r"\b([A-Z]{2,5})\b")

_FACT_CHECK_STOPWORDS = {
    "PE", "EPS", "IPO", "ETF", "CEO", "CFO", "SEC", "NYSE", "GDP", "FED",
    "AI", "ML", "EGF", "ATH", "RSI", "MACD", "IV", "OI",
    "NOT", "AND", "FOR", "THE", "BUT", "USD",
}


def _fact_check_response(
    response_text: str,
    expected_tickers: list[str],
    injected_prices: dict[str, float],
    injected_pe: dict[str, float],
) -> list[str]:
    """Check LLM response for ticker drift and fabricated numbers.

    Returns list of warning strings (empty if all good).
    Compares against live data already in memory — zero LLM cost.
    """
    warnings: list[str] = []

    # 1. Ticker drift: does the response focus on unexpected tickers?
    if expected_tickers:
        expected_set = set(t.upper() for t in expected_tickers)
        mentioned = {m for m in _TICKER_MENTION_RE.findall(response_text) if m not in _FACT_CHECK_STOPWORDS}
        # Only flag if the response predominantly discusses wrong tickers
        unexpected = mentioned - expected_set
        expected_hits = mentioned & expected_set
        if unexpected and not expected_hits:
            warnings.append(f"ticker_drift: response discusses {unexpected} but expected {expected_set}")

    # 2. Price accuracy: compare claimed prices against injected live data
    for ticker, actual_price in injected_prices.items():
        if actual_price is None:
            continue
        # Look for price claims near the ticker mention in the response
        # Simple approach: check all $ claims and see if any are wildly off
        claims = _PRICE_CLAIM_RE.findall(response_text)
        for claim_str in claims:
            try:
                claimed = float(claim_str.replace(",", ""))
                # Only flag if claim is >10% off from any known price (could be buy/sell levels)
                if claimed > 0 and actual_price > 0:
                    pct_off = abs(claimed - actual_price) / actual_price
                    if pct_off > 0.5:  # >50% off — likely fabricated, not a zone/level
                        warnings.append(
                            f"price_suspect: ${claimed:.2f} is {pct_off:.0%} from {ticker} live price ${actual_price:.2f}"
                        )
            except ValueError:
                continue

    return warnings


_PRICE_KEYWORDS = {"price", "trading", "trading at", "cost", "worth", "valued at", "quote", "latest price", "current price"}


def _fetch_index_price_sync(yf_symbol: str) -> float | None:
    """Synchronous yfinance fetch for index price."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(yf_symbol)
        hist = ticker.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
        info = ticker.info or {}
        return info.get("regularMarketPrice") or info.get("currentPrice")
    except Exception as e:
        logger.warning("Index price fetch failed for %s: %s", yf_symbol, e)
        return None


async def _get_index_prices(index_queries: list[tuple[str, str]]) -> str:
    """Fetch live prices for all detected indices and return a formatted context block."""
    if not index_queries:
        return ""

    import asyncio as _asyncio
    tasks = [_asyncio.to_thread(_fetch_index_price_sync, sym) for sym, _ in index_queries]
    prices = await _asyncio.gather(*tasks, return_exceptions=True)

    lines = []
    for (symbol, name), price in zip(index_queries, prices):
        if isinstance(price, Exception) or not price:
            lines.append(f"- {name}: price unavailable")
        else:
            lines.append(f"- {name} (`{symbol}`): **{price:,.2f}** (live, fetched just now)")

    return (
        "## Live Market Data (fetched in real time — use these exact numbers, no approximations)\n"
        + "\n".join(lines)
    )


def _is_price_query(query: str, has_index: bool = False) -> bool:
    q = query.lower()
    if any(kw in q for kw in _PRICE_KEYWORDS):
        return True
    # Short queries that are just an index name are implicitly price questions
    if has_index and len(q.split()) <= 5:
        return True
    return False


def _classify_intent(query: str, has_tickers: bool) -> tuple[str, int]:
    """Classify query intent and return (intent, max_tokens).

    Returns:
        (intent_type, max_output_tokens)
    """
    q = query.lower()

    if any(kw in q for kw in ["portfolio", "holdings", "rebalance", "allocation", "my stocks"]):
        return "portfolio_review", 2000

    if has_tickers:
        return "ticker_analysis", 1500

    persona_lc = settings.analyst_persona.lower()
    if any(kw in q for kw in ["philosophy", "principle", "approach", "methodology", "strategy",
                                "why does", f"what does {persona_lc} think"]):
        return "philosophy", 1000

    return "casual", 400


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    limit = max(1, min(limit, 50))
    last_message_sq = (
        select(
            Message.conversation_id.label("conversation_id"),
            func.max(Message.created_at).label("last_message_at"),
            func.count(Message.id).label("message_count"),
        )
        .group_by(Message.conversation_id)
        .subquery()
    )

    result = await db.execute(
        select(
            Conversation,
            last_message_sq.c.last_message_at,
            last_message_sq.c.message_count,
        )
        .outerjoin(last_message_sq, last_message_sq.c.conversation_id == Conversation.id)
        .where(Conversation.user_id == current_user.id)
        .order_by(func.coalesce(last_message_sq.c.last_message_at, Conversation.created_at).desc())
        .limit(limit)
    )

    conversations = [
        ConversationSummaryResponse(
            id=str(conv.id),
            title=conv.title,
            created_at=conv.created_at,
            last_message_at=last_message_at,
            message_count=message_count or 0,
        )
        for conv, last_message_at, message_count in result.all()
    ]
    return ConversationListResponse(conversations=conversations)


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: str,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    limit = max(1, min(limit, 200))
    try:
        conv_uuid = uuid.UUID(conversation_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc

    conv_result = await db.execute(
        select(Conversation).where(
            Conversation.id == conv_uuid,
            Conversation.user_id == current_user.id,
        )
    )
    conversation = conv_result.scalar_one_or_none()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    total_messages_result = await db.execute(
        select(func.count(Message.id)).where(Message.conversation_id == conv_uuid)
    )
    total_message_count = total_messages_result.scalar() or 0
    role_rank = _message_role_rank()

    message_ids_sq = (
        select(Message.id)
        .where(Message.conversation_id == conv_uuid)
        .order_by(Message.created_at.desc(), role_rank.desc())
        .limit(limit)
        .subquery()
    )
    messages_result = await db.execute(
        select(Message)
        .where(Message.id.in_(select(message_ids_sq.c.id)))
        .order_by(Message.created_at.asc(), role_rank.asc())
    )
    messages = [
        ConversationMessageResponse(
            id=str(msg.id),
            role=msg.role,
            content=msg.content,
            model_used=msg.model_used,
            tokens_used=msg.tokens_used,
            tickers_mentioned=msg.tickers_mentioned,
            metadata_json=msg.metadata_json,
            created_at=msg.created_at,
        )
        for msg in messages_result.scalars().all()
    ]

    return ConversationDetailResponse(
        conversation_id=str(conversation.id),
        title=conversation.title,
        summary=conversation.summary,
        total_message_count=total_message_count,
        truncated=total_message_count > limit,
        messages=messages,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    user_id = current_user.id

    # Admin users bypass rate limiting entirely
    is_allowed, is_degraded = True, False
    limiter: RateLimiter | None = None
    if not current_user.is_admin:
        limiter = RateLimiter(db)
        is_allowed, is_degraded = await limiter.check(user_id)
        if not is_allowed:
            raise HTTPException(
                status_code=429,
                detail="Daily limit reached. Come back tomorrow",
            )

    # Get prev tickers from history (for LLM context in entity extraction)
    prev_tickers: list[str] | None = None
    if request.conversation_id:
        prev_covered, prev_uncovered = await _tickers_from_history(request.conversation_id, db)
        prev_tickers = (prev_covered + prev_uncovered) or None

    # Entity resolution: DB alias cache first, LLM fallback for unknowns
    # LLM also decides: follow-up? general question? social message?
    all_tickers, index_queries, needs_history, is_social = await _extract_entities(
        request.message, db, prev_tickers=prev_tickers,
    )

    # Short-circuit: social/acknowledgment messages — no stock data needed
    if is_social:
        user_created_at = datetime.now(timezone.utc)
        assistant_created_at = user_created_at + timedelta(microseconds=1)
        messages = [
            {"role": "system", "content": "You are a friendly stock market advisor. "
             "The user sent a brief social message. Respond warmly in 1-2 sentences. "
             "Do NOT provide stock analysis or repeat previous answers."},
            {"role": "user", "content": request.message},
        ]
        response = await llm_chat(messages=messages, query=request.message, is_analysis=False, max_tokens=60)
        if limiter:
            await limiter.record_usage(user_id, response.total_tokens)
        conv_id = request.conversation_id
        if not conv_id:
            conv = Conversation(user_id=user_id, title=request.message[:100])
            db.add(conv)
            await db.flush()
            conv_id = str(conv.id)
        db.add(Message(
            conversation_id=uuid.UUID(conv_id),
            role="user",
            content=request.message,
            created_at=user_created_at,
        ))
        db.add(Message(
            conversation_id=uuid.UUID(conv_id),
            role="assistant",
            content=response.content,
            model_used=response.model_used,
            tokens_used=response.total_tokens,
            created_at=assistant_created_at,
        ))
        await db.commit()
        return ChatResponse(reply=response.content, conversation_id=conv_id,
                            model_used=response.model_used, tokens_used=response.total_tokens, is_degraded=is_degraded)

    tickers, uncovered_tickers = await _check_coverage(all_tickers, db)

    # If LLM says this is a follow-up and we found no tickers, carry from history
    if not tickers and not uncovered_tickers and not index_queries and needs_history and prev_tickers:
        tickers, uncovered_tickers = await _check_coverage(prev_tickers, db)

    # Build the list of all tickers for this request
    all_mentioned = list(set(tickers + uncovered_tickers + [sym for sym, _ in index_queries]))

    # Load conversation history with ticker-aware filtering
    history: list[dict] = []
    if request.conversation_id:
        try:
            history = await _load_conversation_history(
                request.conversation_id, db,
                current_tickers=all_mentioned or None,
            )
        except Exception:
            pass

    # Classify intent + set max tokens (uncovered tickers still warrant analysis)
    intent, max_tokens = _classify_intent(request.message, bool(tickers or uncovered_tickers))

    # Build context for covered tickers
    stock_contexts = []
    decision_metrics_strs = []
    injected_prices: dict[str, float] = {}  # for fact-checking

    for ticker in tickers:
        pred = await _get_stock_data(ticker, db)
        if not pred:
            continue

        stock_data_dict = {
            "ticker": pred.ticker,
            "stock_name": pred.stock_name,
            "category": (pred.analyst_labels or {}).get("category", ""),
            "buy_high": float(pred.buy_high) if pred.buy_high else None,
            "buy_low": float(pred.buy_low) if pred.buy_low else None,
            "sell_start": float(pred.sell_start) if pred.sell_start else None,
            "pe_range_high": float(pred.pe_range_high) if pred.pe_range_high else None,
            "pe_range_low": float(pred.pe_range_low) if pred.pe_range_low else None,
            "fair_value": float(pred.fair_value) if pred.fair_value else None,
            "egf": float(pred.egf) if pred.egf else None,
            "egf_direction": float(pred.egf_direction) if pred.egf_direction else None,
            "egf_12m": float(pred.egf_12m) if pred.egf_12m else None,
            "fundamentals": float(pred.fundamentals) if pred.fundamentals else None,
            "trend_status": pred.trend_status,
            "strategy_text": pred.strategy_text,
        }
        stock_contexts.append(format_stock_context(stock_data_dict))

        # Fetch live price + PE
        ticker_symbol = pred.ticker.split()[0]
        price_data = await get_price(ticker_symbol, db)
        current_price = price_data.price
        current_pe = price_data.pe_ratio
        if current_price:
            injected_prices[ticker_symbol.upper()] = current_price

        # Prepend a clear live-price line so the LLM never misses it
        if current_price:
            stock_contexts[-1] = (
                f"**LIVE PRICE: ${current_price:.2f}** (PE: {current_pe:.1f}x)\n"
                if current_pe else
                f"**LIVE PRICE: ${current_price:.2f}**\n"
            ) + stock_contexts[-1]

        # Run decision engine with live price (category is display-only, not a StockData field)
        sd = StockData(**{k: v for k, v in stock_data_dict.items() if k not in ("category",)})
        metrics = analyze_ticker(price=current_price, data=sd, current_pe=current_pe)
        decision_metrics_strs.append(format_metrics(metrics))

    # Load principles (static corpus + active derived principles)
    principles = await _get_principles(db)
    derived = await db.execute(
        select(DerivedPrinciple.principle_text)
        .where(DerivedPrinciple.is_active == True)  # noqa: E712
        .order_by(DerivedPrinciple.confidence_score.desc())
        .limit(20)
    )
    principles.extend(row[0] for row in derived.all())

    # Retrieve relevant analyst chunks (3-channel: ticker SQL + cross-ref + philosophy semantic)
    # Include index symbols (e.g. ^GSPC) so S&P prediction chunks are found
    index_symbols = [sym for sym, _ in index_queries] if index_queries else []
    chunks_with_scores = await _get_relevant_chunks(
        request.message, db, tickers=tickers,
        uncovered_tickers=uncovered_tickers + index_symbols,
    )
    chunk_texts = [_clean_chunk_text(_format_chunk_with_meta(chunk)) for chunk, _ in chunks_with_scores]

    # Update chunk quality tracking
    if chunks_with_scores:
        await _update_chunk_quality(chunks_with_scores, db)

    # Load portfolio context
    portfolio_ctx = await _get_portfolio_context(user_id, db)

    # Build system prompt
    system_prompt = build_system_prompt(
        principles=principles,
        stock_contexts=stock_contexts,
        decision_metrics=decision_metrics_strs,
    )

    # Add retrieved chunks as additional context
    if chunk_texts:
        system_prompt += "\n\n## Relevant Analyst Commentary\n" + "\n---\n".join(chunk_texts)

    # Add portfolio context
    if portfolio_ctx:
        system_prompt += "\n\n" + portfolio_ctx

    # Add earnings calendar
    if tickers:
        earnings = await get_upcoming_earnings(tickers)
        earnings_notice = format_earnings_notice(earnings)
        if earnings_notice:
            system_prompt += "\n\n" + earnings_notice

    # Inject live index prices if any indices mentioned
    if index_queries:
        index_ctx = await _get_index_prices(index_queries)
        if index_ctx:
            system_prompt += "\n\n" + index_ctx

    # For simple price queries, explicitly tell the LLM to lead with the price
    if _is_price_query(request.message, has_index=bool(index_queries)) and (tickers or index_queries):
        system_prompt += (
            "\n\n## Instruction for This Query\n"
            "The user is asking for the current price. "
            "Lead your answer with the LIVE PRICE prominently (it's in the data above), "
            f"then briefly note where that sits relative to {settings.analyst_persona}'s buy/sell zones."
        )

    # Uncovered tickers: fetch live price + PE, apply the analyst's general framework
    if uncovered_tickers:
        persona = settings.analyst_persona
        uncovered_ctx_lines = []
        for unc_ticker in uncovered_tickers:
            try:
                price_data = await get_price(unc_ticker, db)
                if price_data.price:
                    pe_str = f", PE: {price_data.pe_ratio:.1f}x" if price_data.pe_ratio else ""
                    uncovered_ctx_lines.append(
                        f"- **{unc_ticker}**: live price ${price_data.price:.2f}{pe_str}"
                    )
                else:
                    uncovered_ctx_lines.append(f"- **{unc_ticker}**: price unavailable")
            except Exception:
                uncovered_ctx_lines.append(f"- **{unc_ticker}**: price unavailable")

        tickers_str = ", ".join(uncovered_tickers)
        system_prompt += (
            f"\n\n## Tickers Outside {persona}'s Coverage: {tickers_str}\n"
            + "\n".join(uncovered_ctx_lines)
            + f"\n\n{persona} has no specific buy/sell zones or price targets for these. "
            "However, apply his general framework: assess whether the PE ratio is extended "
            "or compressed vs historical norms, whether the price trend is bullish or bearish, "
            "and what the accumulation/distribution logic suggests. "
            f"Be explicit that this is a framework estimate — not a {persona}-specific recommendation."
        )

    # Degradation notice
    degraded_note = ""
    if is_degraded:
        degraded_note = f"\n\n*[Running on the economy engine today — {settings.analyst_persona} says even a sloth knows when to conserve energy.]*"

    # Language mirroring: detect query language and instruct LLM to reply in the same language
    if re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", request.message):
        system_prompt += "\n\n## Language Instruction\nThe user wrote in Chinese (or another CJK language). Reply ENTIRELY in the same language. Do NOT switch to English."
    else:
        system_prompt += "\n\n## Language Instruction\nThe user wrote in English. Reply in English only."

    # Build messages — include conversation history for follow-up context
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)

    # Focus directive: always inject when history exists to prevent re-answering old turns
    if history:
        if all_mentioned:
            messages.append({
                "role": "system",
                "content": f"IMPORTANT: The user is now asking about {', '.join(all_mentioned)}. "
                           f"Answer ONLY about these. Do not re-discuss or re-answer previous questions.",
            })
        else:
            messages.append({
                "role": "system",
                "content": "IMPORTANT: Answer ONLY the user's current question below. "
                           "Do NOT re-answer or summarize any previous questions from the conversation history.",
            })

    messages.append({"role": "user", "content": request.message})

    # Call LLM
    response = await llm_chat(
        messages=messages,
        query=request.message,
        is_analysis=bool(tickers),
        is_degraded=is_degraded,
        max_tokens=max_tokens,
    )

    # Anonymization post-check (Layer 3) — instantiate here so runtime rules are included
    checked = Anonymizer().post_check(response.content)
    if checked.flagged_patterns:
        logger.warning("Anonymization flags in LLM output: %s", checked.flagged_patterns)
    reply = checked.text

    # Fact check: verify ticker focus + number accuracy against live data
    fact_warnings = _fact_check_response(
        reply, all_mentioned, injected_prices, {},
    )
    if fact_warnings:
        logger.warning("Fact check flags: %s", fact_warnings)
        # For ticker drift: prepend a subtle redirect
        for w in fact_warnings:
            if w.startswith("ticker_drift") and all_mentioned:
                reply = f"**Regarding {', '.join(all_mentioned)}:**\n\n" + reply
                break

    reply += degraded_note

    # Record usage (skipped for admin users)
    if limiter:
        await limiter.record_usage(user_id, response.total_tokens)

    # Save conversation + message
    conv_id = request.conversation_id
    if not conv_id:
        conv = Conversation(user_id=user_id, title=request.message[:100])
        db.add(conv)
        await db.flush()
        conv_id = str(conv.id)

    # Save user message + assistant response with annotations
    tickers_for_save = all_mentioned or None
    msg_meta = _classify_message_metadata(request.message, all_mentioned)
    user_created_at = datetime.now(timezone.utc)
    assistant_created_at = user_created_at + timedelta(microseconds=1)

    db.add(Message(
        conversation_id=uuid.UUID(conv_id),
        role="user",
        content=request.message,
        tickers_mentioned=tickers_for_save,
        metadata_json=msg_meta,
        created_at=user_created_at,
    ))
    db.add(Message(
        conversation_id=uuid.UUID(conv_id),
        role="assistant",
        content=reply,
        model_used=response.model_used,
        tokens_used=response.total_tokens,
        tickers_mentioned=tickers_for_save,
        metadata_json=msg_meta,
        created_at=assistant_created_at,
    ))
    await db.commit()

    # Check if conversation is long enough to summarize older turns (background, non-blocking)
    total_msgs_result = await db.execute(
        select(func.count(Message.id)).where(Message.conversation_id == uuid.UUID(conv_id))
    )
    total_msgs = total_msgs_result.scalar() or 0
    if total_msgs >= _SUMMARIZE_AFTER_TURNS * 2 and settings.deepseek_api_key:
        _conv_id = uuid.UUID(conv_id)

        async def _run_summarize() -> None:
            async with async_session() as _db:
                await _summarize_older_messages(_conv_id, _db)

        asyncio.create_task(_run_summarize())

    return ChatResponse(
        reply=reply,
        conversation_id=conv_id,
        model_used=response.model_used,
        tokens_used=response.total_tokens,
        is_degraded=is_degraded,
    )
