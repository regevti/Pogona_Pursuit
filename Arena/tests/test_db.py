import pytest
from alembic import config, script
from alembic.command import stamp, upgrade, revision
from alembic.runtime import migration
import sqlalchemy
from db_models import get_engine
from sqlalchemy_utils import database_exists, create_database


def test_schema_exists():
    engine = get_engine()
    if not database_exists(engine.url):
        create_database(engine.url)
        print(f'Database {config.db_name} was created')


def test_migrations():
    engine = get_engine()
    alembic_cfg = config.Config('alembic.ini')
    script_ = script.ScriptDirectory.from_config(alembic_cfg)
    with engine.begin() as conn:
        context = migration.MigrationContext.configure(conn)
        if not context.get_current_revision():
            stamp(alembic_cfg, 'head')
            revision(alembic_cfg, 'first', True)
            upgrade(alembic_cfg, 'head')
            print('created new stamp and revision for DB')
        assert context.get_current_revision() == script_.get_current_head(), ('Upgrade the database by using:\n'
                                                                              '  alembic revision --autogenerate -m "migration name"\n'
                                                                              '  alembic upgrade head')
