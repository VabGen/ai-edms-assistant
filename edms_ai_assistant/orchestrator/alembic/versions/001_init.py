# orchestrator/alembic/versions/001_init.py
"""
Начальная миграция: схема edms + все таблицы.

Включает:
  - summarization_cache  (перенесено из orchestrator/db/database.py)
  - user_profiles
  - conversation_logs
  - action_history
  - dialog_logs          (feedback-collector)

Revision ID: 001_init
Revises:
Create Date: 2025-01-01 00:00:00
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "001_init"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Схема
    op.execute("CREATE SCHEMA IF NOT EXISTS edms")

    # ── summarization_cache ───────────────────────────────────────────────
    # Перенесено из orchestrator/db/database.py — единственный источник истины
    op.create_table(
        "summarization_cache",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("file_identifier", sa.String(255), nullable=False),
        sa.Column("summary_type", sa.String(50), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "file_identifier", "summary_type", name="_file_summary_uc"
        ),
        schema="edms",
    )
    op.create_index(
        "ix_sumcache_file", "summarization_cache", ["file_identifier"], schema="edms"
    )
    op.create_index(
        "ix_sumcache_type", "summarization_cache", ["summary_type"], schema="edms"
    )

    # ── user_profiles ─────────────────────────────────────────────────────
    op.create_table(
        "user_profiles",
        sa.Column("user_id", sa.String(255), primary_key=True),
        sa.Column("display_name", sa.String(500), nullable=True),
        sa.Column("email", sa.String(500), nullable=True),
        sa.Column("department", sa.String(500), nullable=True),
        sa.Column("role_in_org", sa.String(255), nullable=True),
        sa.Column(
            "preferences",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        schema="edms",
    )

    # ── conversation_logs ─────────────────────────────────────────────────
    op.create_table(
        "conversation_logs",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "user_id",
            sa.String(255),
            sa.ForeignKey("edms.user_profiles.user_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("session_id", sa.String(255), nullable=False),
        sa.Column("role", sa.String(50), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("tool_name", sa.String(255), nullable=True),
        sa.Column(
            "tool_result",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        schema="edms",
    )
    op.create_index("ix_convlogs_user", "conversation_logs", ["user_id"], schema="edms")
    op.create_index("ix_convlogs_session", "conversation_logs", ["session_id"], schema="edms")
    op.create_index("ix_convlogs_ts", "conversation_logs", ["timestamp"], schema="edms")

    # ── action_history ────────────────────────────────────────────────────
    op.create_table(
        "action_history",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "user_id",
            sa.String(255),
            sa.ForeignKey("edms.user_profiles.user_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("action_type", sa.String(255), nullable=False),
        sa.Column("entity_id", sa.String(255), nullable=True),
        sa.Column(
            "action_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("success", sa.Boolean, nullable=False, server_default="true"),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        schema="edms",
    )
    op.create_index("ix_action_user", "action_history", ["user_id"], schema="edms")
    op.create_index("ix_action_type", "action_history", ["action_type"], schema="edms")
    op.create_index("ix_action_ts", "action_history", ["timestamp"], schema="edms")

    # ── dialog_logs (feedback-collector) ──────────────────────────────────
    op.create_table(
        "dialog_logs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("session_id", sa.String(255), nullable=False),
        sa.Column("user_query", sa.Text, nullable=False),
        sa.Column("normalized_query", sa.Text, nullable=True),
        sa.Column("intent", sa.String(100), nullable=True),
        sa.Column("confidence", sa.Float, nullable=True),
        sa.Column(
            "entities",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("selected_tool", sa.String(255), nullable=True),
        sa.Column(
            "tool_args",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "tool_result",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("final_response", sa.Text, nullable=True),
        sa.Column("model_used", sa.String(100), nullable=True),
        sa.Column("tokens_used", sa.Integer, nullable=False, server_default="0"),
        sa.Column("user_feedback", sa.Integer, nullable=True),
        sa.Column("feedback_comment", sa.Text, nullable=True),
        sa.Column("bypass_llm", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("latency_ms", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("feedback_at", sa.DateTime(timezone=True), nullable=True),
        schema="edms",
    )
    op.create_index("ix_dialogs_user", "dialog_logs", ["user_id"], schema="edms")
    op.create_index("ix_dialogs_intent", "dialog_logs", ["intent"], schema="edms")
    op.create_index("ix_dialogs_ts", "dialog_logs", ["created_at"], schema="edms")


def downgrade() -> None:
    for table in [
        "dialog_logs",
        "action_history",
        "conversation_logs",
        "user_profiles",
        "summarization_cache",
    ]:
        op.drop_table(table, schema="edms")
    op.execute("DROP SCHEMA IF EXISTS edms CASCADE")