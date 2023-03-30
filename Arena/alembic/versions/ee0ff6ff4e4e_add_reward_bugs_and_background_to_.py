"""add reward_bugs and background to AnimalID

Revision ID: ee0ff6ff4e4e
Revises: b16990172ba5
Create Date: 2022-12-03 17:13:27.338457

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'ee0ff6ff4e4e'
down_revision = 'b16990172ba5'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('animals', sa.Column('reward_bugs', sa.String(), nullable=True))
    op.add_column('animals', sa.Column('background_color', sa.String(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('animals', 'background_color')
    op.drop_column('animals', 'reward_bugs')
    # ### end Alembic commands ###
