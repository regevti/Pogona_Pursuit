"""animal_id to videos

Revision ID: cf9fff890dee
Revises: 650f2591a8b3
Create Date: 2022-11-08 09:10:56.341744

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'cf9fff890dee'
down_revision = '650f2591a8b3'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('videos', sa.Column('animal_id', sa.String(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('videos', 'animal_id')
    # ### end Alembic commands ###
