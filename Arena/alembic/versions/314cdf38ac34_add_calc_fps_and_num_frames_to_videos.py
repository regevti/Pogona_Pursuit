"""add calc_fps and num_frames to videos

Revision ID: 314cdf38ac34
Revises: 09f89c5d0d09
Create Date: 2022-10-11 10:34:46.002087

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '314cdf38ac34'
down_revision = '09f89c5d0d09'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('videos', sa.Column('start_time', sa.DateTime(), nullable=True))
    op.add_column('videos', sa.Column('calc_fps', sa.Float(), nullable=True))
    op.add_column('videos', sa.Column('num_frames', sa.Integer(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('videos', 'num_frames')
    op.drop_column('videos', 'calc_fps')
    op.drop_column('videos', 'start_time')
    # ### end Alembic commands ###