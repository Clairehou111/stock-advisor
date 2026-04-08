"""V1 schema enhancements: analyst_chunks temporal fields + derived_principles table

Revision ID: 001_v1_schema
Revises:
Create Date: 2026-04-06
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

revision = "001_v1_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create all base tables first (initial migration)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "users",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("username", sa.String(100), unique=True, nullable=True),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("daily_token_limit", sa.Integer, default=100_000),
        sa.Column("is_admin", sa.Boolean, default=False),
        sa.Column("preferred_lang", sa.String(10), default="en"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "conversations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(255)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "messages",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("conversation_id", UUID(as_uuid=True), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("model_used", sa.String(50)),
        sa.Column("tokens_used", sa.Integer),
        sa.Column("source_ids", ARRAY(UUID(as_uuid=True))),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "portfolio_holdings",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("ticker", sa.String(20), nullable=False),
        sa.Column("shares", sa.Numeric(12, 4), nullable=False),
        sa.Column("avg_cost_basis", sa.Numeric(12, 4)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("user_id", "ticker", name="uq_user_ticker"),
    )

    op.create_table(
        "upload_sources",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("file_type", sa.String(20), nullable=False),
        sa.Column("r2_key", sa.String(500)),
        sa.Column("sheet_name", sa.String(100)),
        sa.Column("extracted_json", JSONB),
        sa.Column("conflict_report", JSONB),
        sa.Column("change_summary", sa.Text),
        sa.Column("upload_timestamp", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "stock_predictions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("ticker", sa.String(20), nullable=False, index=True),
        sa.Column("stock_name", sa.String(100)),
        sa.Column("buy_high", sa.Numeric(12, 4)),
        sa.Column("buy_low", sa.Numeric(12, 4)),
        sa.Column("sell_start", sa.Numeric(12, 4)),
        sa.Column("pe_range_high", sa.Numeric(8, 2)),
        sa.Column("pe_range_low", sa.Numeric(8, 2)),
        sa.Column("fair_value", sa.Numeric(12, 4)),
        sa.Column("egf", sa.Numeric(8, 4)),
        sa.Column("egf_direction", sa.Numeric(8, 4)),
        sa.Column("egf_12m", sa.Numeric(8, 4)),
        sa.Column("fundamentals", sa.Numeric(4, 1)),
        sa.Column("trend_status", sa.String(255)),
        sa.Column("prob_new_ath", sa.Numeric(4, 2)),
        sa.Column("strategy_text", sa.Text),
        sa.Column("analyst_labels", JSONB),
        sa.Column("upload_source_id", UUID(as_uuid=True), sa.ForeignKey("upload_sources.id")),
        sa.Column("is_current", sa.Boolean, default=True, index=True),
        sa.Column("superseded_by", UUID(as_uuid=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_stock_predictions_ticker_current", "stock_predictions", ["ticker", "is_current"])

    op.create_table(
        "analyst_chunks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("upload_source_id", UUID(as_uuid=True), sa.ForeignKey("upload_sources.id")),
        sa.Column("ticker", sa.String(20), index=True),
        sa.Column("chunk_type", sa.String(50), nullable=False),
        sa.Column("content_text", sa.Text, nullable=False),
        sa.Column("embedding", Vector(768)),
        sa.Column("temporal_scope", sa.String(20)),
        sa.Column("metadata_json", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        # V1 temporal fields
        sa.Column("outlook_horizon", sa.String(20)),
        sa.Column("publish_date", sa.Date),
        sa.Column("tickers_mentioned", ARRAY(sa.String(20))),
        sa.Column("thesis_direction", sa.String(20)),
        # V1 quality tracking
        sa.Column("retrieval_count", sa.Integer, default=0),
        sa.Column("avg_relevance", sa.Numeric(4, 3)),
        sa.Column("last_retrieved", sa.DateTime(timezone=True)),
        sa.Column("is_stale", sa.Boolean, default=False),
    )

    op.create_table(
        "principle_corpus",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("principle_text", sa.Text, nullable=False),
        sa.Column("category", sa.String(50), nullable=False, index=True),
        sa.Column("source_ids", ARRAY(UUID(as_uuid=True))),
        sa.Column("version", sa.Integer, default=1),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "derived_principles",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("principle_text", sa.Text, nullable=False),
        sa.Column("category", sa.String(50), nullable=False, index=True),
        sa.Column("confidence_score", sa.Numeric(4, 3), default=0.3),
        sa.Column("times_stated", sa.Integer, default=1),
        sa.Column("source_chunk_ids", ARRAY(UUID(as_uuid=True))),
        sa.Column("first_seen", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("last_reinforced", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("is_active", sa.Boolean, default=True, index=True),
        sa.Column("superseded_by", UUID(as_uuid=True)),
    )

    op.create_table(
        "rate_limit_usage",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("usage_date", sa.Date, nullable=False),
        sa.Column("tokens_used", sa.Integer, default=0),
        sa.Column("queries_count", sa.Integer, default=0),
        sa.UniqueConstraint("user_id", "usage_date", name="uq_user_date"),
    )

    op.create_table(
        "anonymization_rules",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("original_term", sa.String(500), unique=True, nullable=False),
        sa.Column("replacement", sa.String(500), nullable=False),
        sa.Column("category", sa.String(50), nullable=False),
    )

    op.create_table(
        "price_cache",
        sa.Column("ticker", sa.String(20), primary_key=True),
        sa.Column("price", sa.Numeric(12, 4), nullable=False),
        sa.Column("pe_ratio", sa.Numeric(8, 2)),
        sa.Column("fetched_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("price_cache")
    op.drop_table("anonymization_rules")
    op.drop_table("rate_limit_usage")
    op.drop_table("derived_principles")
    op.drop_table("principle_corpus")
    op.drop_table("analyst_chunks")
    op.drop_table("stock_predictions")
    op.drop_table("upload_sources")
    op.drop_table("portfolio_holdings")
    op.drop_table("messages")
    op.drop_table("conversations")
    op.drop_table("users")
