"""add users.provider_api_keys (extra text-provider keys)

Adds a JSON column that stores encrypted API keys for the extra text providers
(Mistral / Groq / OpenRouter) used as failover for the drug assistant, shaped
as ``{provider: [enc_key, ...]}`` with up to 10 slots per provider.

Revision ID: 0003_add_provider_api_keys
Revises: 0002_add_user_queries
Create Date: 2026-07-18
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0003_add_provider_api_keys"
down_revision = "0002_add_user_queries"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("provider_api_keys", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "provider_api_keys")
