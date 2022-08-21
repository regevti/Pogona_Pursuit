from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, Boolean, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy_utils import database_exists, create_database
import config


Base = declarative_base()


class Experiment(Base):
    __tablename__ = 'experiments'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    start_time = Column(DateTime)
    end_time = Column(DateTime, nullable=True, default=None)
    animal_id = Column(Integer)
    time_between_blocks = Column(Integer)
    experiment_path = Column(String)
    blocks = relationship('Block')


class Block(Base):
    __tablename__ = 'blocks'

    id = Column(Integer, primary_key=True)
    start_time = Column(DateTime)
    end_time = Column(DateTime, nullable=True, default=None)
    experiment_id = Column(Integer, ForeignKey('experiments.id'))
    block_id = Column(Integer)


class Trial(Base):
    __tablename__ = 'trials'

    id = Column(Integer, primary_key=True)
    block_id = Column(Integer, ForeignKey('blocks.id'))
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    duration = Column(Float)
    bug_type = Column(String)
    trial_id = Column(Integer)


class Temperature(Base):
    __tablename__ = 'temperatures'

    id = Column(Integer, primary_key=True)
    time = Column(DateTime)
    value = Column(Float)


class Strikes(Base):
    __tablename__ = 'strikes'

    id = Column(Integer, primary_key=True)
    time = Column(DateTime)
    is_hit = Column(Boolean)
    x = Column(Float)
    y = Column(Float)
    bug_type = Column(String)
    experiment_id = Column(Integer, ForeignKey('experiments.id'))
    block_id = Column(Integer, ForeignKey('blocks.id'), nullabel=True)
    trial_id = Column(Integer, ForeignKey('trials.id'), nullable=True)


class BugPosition(Base):
    __tablename__ = 'bug_positions'

    id = Column(Integer, primary_key=True)
    time = Column(DateTime)
    x = Column(Float)
    y = Column(Float)
    trial_id = Column(Integer, ForeignKey('trials.id'))


class TrialsTimes(Base):
    __tablename__ = 'bug_positions'

    id = Column(Integer, primary_key=True)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    duration = Column(Float)
    bug_type = Column(String)
    trial_id = Column(Integer)


class ORM:
    def __init__(self):
        self.engine = get_engine()
        self.session = sessionmaker(bind=self.engine)
        self.experiment_model = None
        self.blocks = []

    def commit_experiment(self, exp, blocks):
        with self.session() as s:
            kwargs = {c.name: getattr(exp, c.name)
                      for c in Experiment.__table__.columns if c not in ['id', 'end_time']}
            self.experiment_model = Experiment(**kwargs)
            s.add(self.experiment_model)
            s.commit()
            for blk in blocks:
                self.commit_block(s, blk)

    def commit_block(self, s, blk):
        kwargs = {c.name: getattr(self, c.name)
                  for c in blk.__table__.columns if c not in ['id', 'end_time'] and not c.foreign_keys}
        kwargs['experiment_id'] = self.experiment_model.id
        s.add(blk(**kwargs))
        s.commit()

    def update_block_time(self, block_id):
        with self.session() as s:
            block_model = s.query(Block).filter_by(block_id=block_id, experiment_id=self.experiment_model.id).first()
            block_model.update({'end_time': 1})

    def update_experiment_time(self):
        with self.session() as s:
            self.experiment_model.update({'end_time': 1})


def get_engine():
    return create_engine(config.sqlalchemy_url)


if __name__ == '__main__':
    # create all models
    engine = get_engine()
    if not database_exists(engine.url):
        print(f'Database {config.db_name} was created')
        create_database(engine.url)

    # Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    # run "alembic stamp head" to create a versions table in the db

    # Updating
    # If you change something in SQLAlchemy models, you need to create a migration file using:
    # alembic revision --autogenerate -m "migration name"
    #
    # and then to upgrade (make sure there are no open sessions before):
    # alembic upgrade head
