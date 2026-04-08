import uuid
from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, generate_uuid


# ── Users ──────────────────────────────────────────────────────────────────────


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    daily_token_limit: Mapped[int] = mapped_column(Integer, default=100_000)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    preferred_lang: Mapped[str] = mapped_column(String(10), default="en")

    conversations: Mapped[list["Conversation"]] = relationship(back_populates="user")
    portfolio_holdings: Mapped[list["PortfolioHolding"]] = relationship(back_populates="user")


# ── Conversations & Messages ───────────────────────────────────────────────────


class Conversation(Base, TimestampMixin):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    title: Mapped[str | None] = mapped_column(String(255))
    summary: Mapped[str | None] = mapped_column(Text)           # rolling summary of older turns
    summarized_through: Mapped[int] = mapped_column(Integer, default=0)  # count of messages summarized
    context_map: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # per-ticker rolling summary

    user: Mapped["User"] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(back_populates="conversation", order_by="Message.created_at")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    conversation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(20))  # "user" | "assistant" | "system"
    content: Mapped[str] = mapped_column(Text)
    model_used: Mapped[str | None] = mapped_column(String(50))
    tokens_used: Mapped[int | None] = mapped_column(Integer)
    source_ids: Mapped[list | None] = mapped_column(ARRAY(UUID(as_uuid=True)), nullable=True)
    tickers_mentioned: Mapped[list | None] = mapped_column(ARRAY(String(20)), nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


# ── Portfolio ──────────────────────────────────────────────────────────────────


class PortfolioHolding(Base, TimestampMixin):
    __tablename__ = "portfolio_holdings"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    ticker: Mapped[str] = mapped_column(String(20))
    shares: Mapped[float] = mapped_column(Numeric(12, 4))
    avg_cost_basis: Mapped[float | None] = mapped_column(Numeric(12, 4))

    user: Mapped["User"] = relationship(back_populates="portfolio_holdings")

    __table_args__ = (UniqueConstraint("user_id", "ticker", name="uq_user_ticker"),)


# ── Stock Predictions (structured facts from Excel + documents) ────────────────


class StockPrediction(Base, TimestampMixin):
    __tablename__ = "stock_predictions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    stock_name: Mapped[str | None] = mapped_column(String(100))

    # Buying range
    buy_high: Mapped[float | None] = mapped_column(Numeric(12, 4))
    buy_low: Mapped[float | None] = mapped_column(Numeric(12, 4))

    # Sell start (first trim level — surfaced as threshold, not exact order price)
    sell_start: Mapped[float | None] = mapped_column(Numeric(12, 4))

    # PE range
    pe_range_high: Mapped[float | None] = mapped_column(Numeric(8, 2))
    pe_range_low: Mapped[float | None] = mapped_column(Numeric(8, 2))

    # Fair value
    fair_value: Mapped[float | None] = mapped_column(Numeric(12, 4))

    # Earnings Growth Factor
    egf: Mapped[float | None] = mapped_column(Numeric(8, 4))
    egf_direction: Mapped[float | None] = mapped_column(Numeric(8, 4))
    egf_12m: Mapped[float | None] = mapped_column(Numeric(8, 4))

    # Fundamentals & trend
    fundamentals: Mapped[float | None] = mapped_column(Numeric(4, 1))
    trend_status: Mapped[str | None] = mapped_column(String(255))
    prob_new_ath: Mapped[float | None] = mapped_column(Numeric(4, 2))

    # Strategy
    strategy_text: Mapped[str | None] = mapped_column(Text)
    analyst_labels: Mapped[dict | None] = mapped_column(JSONB)

    # Tracking
    upload_source_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("upload_sources.id"))
    is_current: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    superseded_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))

    __table_args__ = (
        Index("ix_stock_predictions_ticker_current", "ticker", "is_current"),
    )


# ── Analyst Chunks (semantic layer for RAG) ────────────────────────────────────


class AnalystChunk(Base):
    __tablename__ = "analyst_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    upload_source_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("upload_sources.id"))
    ticker: Mapped[str | None] = mapped_column(String(20), index=True)
    chunk_type: Mapped[str] = mapped_column(String(50))  # philosophy / prediction / commentary / egf_explanation
    content_text: Mapped[str] = mapped_column(Text)
    embedding = mapped_column(Vector(768), nullable=True)  # Gemini text-embedding-004 = 768 dims
    temporal_scope: Mapped[str | None] = mapped_column(String(20))  # short_term / long_term / general
    metadata_json: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # V1 temporal fields
    outlook_horizon: Mapped[str | None] = mapped_column(String(20))  # 1_month / 3_month / 6_month / multi_year
    publish_date: Mapped[date | None] = mapped_column(Date)
    tickers_mentioned: Mapped[list | None] = mapped_column(ARRAY(String(20)))
    thesis_direction: Mapped[str | None] = mapped_column(String(20))  # bullish / bearish / neutral / mixed

    # V1 quality tracking
    retrieval_count: Mapped[int] = mapped_column(Integer, default=0)
    avg_relevance: Mapped[float | None] = mapped_column(Numeric(4, 3))
    last_retrieved: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_stale: Mapped[bool] = mapped_column(Boolean, default=False)


# ── Principle Corpus ───────────────────────────────────────────────────────────


class PrincipleCorpus(Base, TimestampMixin):
    __tablename__ = "principle_corpus"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    principle_text: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(50), index=True)  # valuation / accumulation / distribution / risk / sentiment
    source_ids: Mapped[list | None] = mapped_column(ARRAY(UUID(as_uuid=True)))
    version: Mapped[int] = mapped_column(Integer, default=1)


# ── Derived Principles (evolves from document ingestion) ─────────────────────


class DerivedPrinciple(Base):
    __tablename__ = "derived_principles"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    principle_text: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(50), index=True)  # valuation / accumulation / distribution / risk / sentiment
    confidence_score: Mapped[float] = mapped_column(Numeric(4, 3), default=0.3)
    times_stated: Mapped[int] = mapped_column(Integer, default=1)
    source_chunk_ids: Mapped[list | None] = mapped_column(ARRAY(UUID(as_uuid=True)))
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_reinforced: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    superseded_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))


# ── Upload Sources ─────────────────────────────────────────────────────────────


class UploadSource(Base):
    __tablename__ = "upload_sources"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    file_type: Mapped[str] = mapped_column(String(20))  # xlsx / image / text / pdf / patreon
    r2_key: Mapped[str | None] = mapped_column(String(500))
    sheet_name: Mapped[str | None] = mapped_column(String(100))
    extracted_json: Mapped[dict | None] = mapped_column(JSONB)
    conflict_report: Mapped[dict | None] = mapped_column(JSONB)
    change_summary: Mapped[str | None] = mapped_column(Text)
    raw_content: Mapped[dict | None] = mapped_column(JSONB)  # original unprocessed content
    upload_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ── Trade Signals (structured buy/trim/sell signals from posts) ───────────────


class TradeSignal(Base):
    __tablename__ = "trade_signals"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    action: Mapped[str] = mapped_column(String(10))  # buy / trim / sell / watch
    price_level: Mapped[float | None] = mapped_column(Numeric(12, 4))
    confidence: Mapped[str | None] = mapped_column(String(20))  # high / medium / low
    post_id: Mapped[str | None] = mapped_column(String(50), index=True)  # patreon post id
    publish_date: Mapped[date | None] = mapped_column(Date, index=True)
    context_text: Mapped[str | None] = mapped_column(Text)  # sentence it came from
    upload_source_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("upload_sources.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ── Rate Limit Usage ───────────────────────────────────────────────────────────


class RateLimitUsage(Base):
    __tablename__ = "rate_limit_usage"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    usage_date: Mapped[date] = mapped_column(Date)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    queries_count: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (UniqueConstraint("user_id", "usage_date", name="uq_user_date"),)


# ── Anonymization Rules ───────────────────────────────────────────────────────


class AnonymizationRule(Base):
    __tablename__ = "anonymization_rules"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    original_term: Mapped[str] = mapped_column(String(500), unique=True)
    replacement: Mapped[str] = mapped_column(String(500))
    category: Mapped[str] = mapped_column(String(50))  # name / url / nickname / platform


# ── Price Cache ────────────────────────────────────────────────────────────────


class PriceCache(Base):
    __tablename__ = "price_cache"

    ticker: Mapped[str] = mapped_column(String(20), primary_key=True)
    price: Mapped[float] = mapped_column(Numeric(12, 4))
    pe_ratio: Mapped[float | None] = mapped_column(Numeric(8, 2))
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ── Ingest Tasks ─────────────────────────────────────────────────────────────


class IngestTask(Base):
    __tablename__ = "ingest_tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)          # UUID string
    status: Mapped[str] = mapped_column(String(20), default="running")     # running / done / error
    messages: Mapped[list] = mapped_column(JSONB, default=list)
    result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ── Entity Alias Cache ────────────────────────────────────────────────────────
# Maps user-typed aliases (typos, nicknames, shorthands) → canonical symbols.
# Seeded with known aliases on startup; grows automatically as LLM resolves new ones.


class EntityAlias(Base):
    __tablename__ = "entity_aliases"

    alias: Mapped[str] = mapped_column(String(200), primary_key=True)  # lowercase, e.g. "sp", "fruit stock"
    resolved_type: Mapped[str] = mapped_column(String(10))             # "ticker" | "index"
    resolved_value: Mapped[str] = mapped_column(String(20))            # "AAPL" | "^GSPC"
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
