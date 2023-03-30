"""add arena_name

Revision ID: d1e35203f184
Revises: 3e869a7cb9e3
Create Date: 2023-03-28 10:44:35.838608

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd1e35203f184'
down_revision = '3e869a7cb9e3'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('animals', sa.Column('arena', sa.String(), nullable=True))
    op.add_column('experiments', sa.Column('arena', sa.String(), nullable=True))
    op.add_column('temperatures', sa.Column('arena', sa.String(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('temperatures', 'arena')
    op.drop_column('experiments', 'arena')
    op.drop_column('animals', 'arena')
    # ### end Alembic commands ###
