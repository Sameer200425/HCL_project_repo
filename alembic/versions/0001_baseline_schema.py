# pyright: reportMissingImports=false, reportAttributeAccessIssue=false

"""Baseline schema migration.

Revision ID: 0001_baseline_schema
Revises: 
Create Date: 2026-04-04
"""

from alembic import op
import sqlalchemy as sa


revision = "0001_baseline_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("username", sa.String(length=100), nullable=False),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("full_name", sa.String(length=255), nullable=True),
        sa.Column("role", sa.String(length=50), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_users_id", "users", ["id"], unique=False)
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_username", "users", ["username"], unique=True)

    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("filename", sa.String(length=255), nullable=True),
        sa.Column("image_hash", sa.String(length=64), nullable=True),
        sa.Column("model_name", sa.String(length=50), nullable=False),
        sa.Column("predicted_class", sa.String(length=50), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("probabilities", sa.JSON(), nullable=True),
        sa.Column("risk_level", sa.String(length=20), nullable=False),
        sa.Column("inference_time_ms", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_predictions_id", "predictions", ["id"], unique=False)

    op.create_table(
        "api_keys",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("last_used_at", sa.DateTime(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_api_keys_id", "api_keys", ["id"], unique=False)
    op.create_index("ix_api_keys_key", "api_keys", ["key"], unique=True)

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("action", sa.String(length=100), nullable=False),
        sa.Column("resource_type", sa.String(length=50), nullable=True),
        sa.Column("resource_id", sa.Integer(), nullable=True),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column("ip_address", sa.String(length=45), nullable=True),
        sa.Column("user_agent", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_audit_logs_id", "audit_logs", ["id"], unique=False)

    op.create_table(
        "model_metrics",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("model_name", sa.String(length=50), nullable=False),
        sa.Column("date", sa.DateTime(), nullable=True),
        sa.Column("total_predictions", sa.Integer(), nullable=True),
        sa.Column("avg_confidence", sa.Float(), nullable=True),
        sa.Column("avg_inference_time_ms", sa.Float(), nullable=True),
        sa.Column("genuine_count", sa.Integer(), nullable=True),
        sa.Column("fraud_count", sa.Integer(), nullable=True),
        sa.Column("tampered_count", sa.Integer(), nullable=True),
        sa.Column("forged_count", sa.Integer(), nullable=True),
    )
    op.create_index("ix_model_metrics_id", "model_metrics", ["id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_model_metrics_id", table_name="model_metrics")
    op.drop_table("model_metrics")
    op.drop_index("ix_audit_logs_id", table_name="audit_logs")
    op.drop_table("audit_logs")
    op.drop_index("ix_api_keys_key", table_name="api_keys")
    op.drop_index("ix_api_keys_id", table_name="api_keys")
    op.drop_table("api_keys")
    op.drop_index("ix_predictions_id", table_name="predictions")
    op.drop_table("predictions")
    op.drop_index("ix_users_username", table_name="users")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_index("ix_users_id", table_name="users")
    op.drop_table("users")
