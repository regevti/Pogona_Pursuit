import json
from types import FunctionType
from functools import wraps
import numpy as np
from datetime import datetime, timedelta, date
from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, Boolean, create_engine, cast, Date
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy_utils import database_exists, create_database
from sqlalchemy.dialects.postgresql import JSON
import config
from cache import RedisCache, CacheColumns as cc
from loggers import get_logger

Base = declarative_base()


class Animal(Base):
    __tablename__ = 'animals'

    id = Column(Integer, primary_key=True)
    animal_id = Column(String)
    start_time = Column(DateTime)
    end_time = Column(DateTime, nullable=True)
    sex = Column(String)
    bug_types = Column(String)
    reward_bugs = Column(String)
    background_color = Column(String)
    reward_any_touch_prob = Column(Float, default=0)
    exit_hole = Column(String, nullable=True)
    audit = relationship('AnimalSettingsHistory')


class AnimalSettingsHistory(Base):
    __tablename__ = 'animal_settings'

    id = Column(Integer, primary_key=True)
    time = Column(DateTime)
    animal_id = Column(String)
    sex = Column(String)
    bug_types = Column(String)
    reward_bugs = Column(String)
    background_color = Column(String)
    reward_any_touch_prob = Column(Float, default=0)
    exit_hole = Column(String, nullable=True)
    animal_id_key = Column(Integer, ForeignKey('animals.id'))


class Schedule(Base):
    __tablename__ = 'schedules'

    id = Column(Integer, primary_key=True)
    date = Column(DateTime)
    animal_id = Column(String)
    experiment_name = Column(String)


class ModelGroup(Base):
    __tablename__ = 'model_groups'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    versions = relationship('ModelVersion')


class ModelVersion(Base):
    __tablename__ = 'model_versions'

    id = Column(Integer, primary_key=True)
    create_date = Column(DateTime)
    version = Column(String)
    folder = Column(String)
    model_group_id = Column(Integer, ForeignKey('model_groups.id'))
    is_active = Column(Boolean, default=True)


class Experiment(Base):
    __tablename__ = 'experiments'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    start_time = Column(DateTime)
    end_time = Column(DateTime, nullable=True, default=None)
    animal_id = Column(String)
    cameras = Column(String)  # list
    num_blocks = Column(Integer)
    extra_time_recording = Column(Integer)
    time_between_blocks = Column(Integer)
    experiment_path = Column(String)
    blocks = relationship('Block')


class Block(Base):
    __tablename__ = 'blocks'

    id = Column(Integer, primary_key=True)
    start_time = Column(DateTime)
    end_time = Column(DateTime, nullable=True, default=None)
    experiment_id = Column(Integer, ForeignKey('experiments.id'))
    block_id = Column(Integer)  # the ID of the block inside the experiment
    num_trials = Column(Integer)
    trial_duration = Column(Integer)
    iti = Column(Integer)
    block_type = Column(String)
    bug_types = Column(String)  # originally a list
    bug_speed = Column(Integer)
    bug_size = Column(Integer)
    is_default_bug_size = Column(Boolean)
    exit_hole = Column(String)
    reward_type = Column(String)
    reward_bugs = Column(String)  # originally a list
    reward_any_touch_prob = Column(Float)
    media_url = Column(String, nullable=True)
    movement_type = Column(String)
    is_anticlockwise = Column(Boolean)
    target_drift = Column(String, nullable=True)
    bug_height = Column(Integer)
    time_between_bugs = Column(Integer)
    background_color = Column(String, nullable=True)
    strikes = relationship('Strike')
    trials = relationship('Trial')
    videos = relationship('Video')


class Trial(Base):
    """Trial refers to a single trajectory of the bug/media on the screen"""
    __tablename__ = 'trials'

    id = Column(Integer, primary_key=True)
    in_block_trial_id = Column(Integer)  # The ID of the trial inside the block. Starts from 1
    block_id = Column(Integer, ForeignKey('blocks.id'))
    start_time = Column(DateTime)
    end_time = Column(DateTime, nullable=True, default=None)
    duration = Column(Float, nullable=True, default=None)
    bug_trajectory = Column(JSON, nullable=True)
    strikes = relationship('Strike')


class Temperature(Base):
    __tablename__ = 'temperatures'

    id = Column(Integer, primary_key=True)
    time = Column(DateTime)
    value = Column(Float)
    block_id = Column(Integer, ForeignKey('blocks.id'), nullable=True)


class Strike(Base):
    __tablename__ = 'strikes'

    id = Column(Integer, primary_key=True)
    time = Column(DateTime)
    is_hit = Column(Boolean)
    is_reward_bug = Column(Boolean)
    is_climbing = Column(Boolean)
    x = Column(Float)
    y = Column(Float)
    bug_x = Column(Float)
    bug_y = Column(Float)
    bug_type = Column(String)
    bug_size = Column(Integer)
    in_block_trial_id = Column(Integer, nullable=True)  # The ID of the trial inside the block. Starts from 1
    prediction_distance = Column(Float, nullable=True)
    calc_speed = Column(Float, nullable=True)
    projected_strike_coords = Column(JSON, nullable=True)
    projected_leap_coords = Column(JSON, nullable=True)
    max_acceleration = Column(Float, nullable=True)
    strike_frame = Column(Integer, nullable=True)
    leap_frame = Column(Integer, nullable=True)
    block_id = Column(Integer, ForeignKey('blocks.id'), nullable=True)
    trial_id = Column(Integer, ForeignKey('trials.id'), nullable=True)
    video_id = Column(Integer, ForeignKey('videos.id'), nullable=True)


class Video(Base):
    __tablename__ = 'videos'

    id = Column(Integer, primary_key=True)
    cam_name = Column(String)
    path = Column(String)
    start_time = Column(DateTime, nullable=True)
    fps = Column(Float)
    calc_fps = Column(Float, nullable=True)
    num_frames = Column(Integer, nullable=True)
    animal_id = Column(String, nullable=True)
    frames = Column(JSON, nullable=True)
    compression_status = Column(Integer, default=0)  # 0: no compression, 1: compressed, 2: error
    block_id = Column(Integer, ForeignKey('blocks.id'), nullable=True)
    predictions = relationship('VideoPrediction')


class VideoPrediction(Base):
    __tablename__ = 'video_predictions'

    id = Column(Integer, primary_key=True)
    predictor_name = Column(String)
    start_time = Column(DateTime)
    data = Column(JSON)
    video_id = Column(Integer, ForeignKey('videos.id'), nullable=True)


class PoseEstimation(Base):
    __tablename__ = 'pose_estimations'

    id = Column(Integer, primary_key=True)
    cam_name = Column(String)
    start_time = Column(DateTime)
    x = Column(Float)
    y = Column(Float)
    model = Column(String, nullable=True)
    animal_id = Column(String, nullable=True)
    angle = Column(Float, nullable=True)
    engagement = Column(Float, nullable=True)
    video_id = Column(Integer, ForeignKey('videos.id'), nullable=True)
    block_id = Column(Integer, ForeignKey('blocks.id'), nullable=True)


class Reward(Base):
    __tablename__ = 'rewards'

    id = Column(Integer, primary_key=True)
    time = Column(DateTime)
    animal_id = Column(String)
    block_id = Column(Integer, ForeignKey('blocks.id'), nullable=True)


def commit_func(method):
    @wraps(method)
    def wrapped(*args, **kwargs):
        if config.DISABLE_DB:
            return
        return method(*args, **kwargs)
    return wrapped


ANIMAL_SETTINGS_LISTS = ['bug_types', 'reward_bugs']


class ORM:
    def __init__(self):
        self.engine = get_engine()
        self.session = sessionmaker(bind=self.engine)
        self.current_experiment_id = None
        self.cache = RedisCache()
        self.logger = get_logger('orm')

    @commit_func
    def commit_experiment(self, exp):
        with self.session() as s:
            kwargs = {c.name: getattr(exp, c.name)
                      for c in Experiment.__table__.columns if c.name not in ['id', 'end_time', 'cameras']}
            exp_model = Experiment(**kwargs)
            exp_model.cameras = ','.join(list(exp.cameras.keys()))
            s.add(exp_model)
            s.commit()
            self.current_experiment_id = exp_model.id

    @commit_func
    def commit_block(self, blk, is_cache_set=True):
        with self.session() as s:
            kwargs = {c.name: getattr(blk, c.name)
                      for c in Block.__table__.columns if c.name not in ['id', 'end_time']
                      and not c.foreign_keys}
            kwargs['experiment_id'] = self.current_experiment_id
            for k in ['reward_bugs', 'bug_types']:  # convert lists to strings
                if isinstance(kwargs[k], list):
                    kwargs[k] = ','.join(kwargs[k])
            b = Block(**kwargs)
            s.add(b)
            s.commit()
            block_id = b.id
            if is_cache_set:
                self.cache.set(cc.CURRENT_BLOCK_DB_INDEX, block_id)
        return block_id

    @commit_func
    def commit_trial(self, trial_dict):
        kwargs = {c.name: trial_dict.get(c.name)
                  for c in Trial.__table__.columns if c.name not in ['id'] and not c.foreign_keys}
        kwargs['block_id'] = trial_dict.get('block_id') or self.cache.get(cc.CURRENT_BLOCK_DB_INDEX)
        with self.session() as s:
            trial = Trial(**kwargs)
            s.add(trial)
            s.commit()
            trial_id = trial.id
        return trial_id

    @commit_func
    def update_trial_data(self, trial_dict):
        trial_id = trial_dict.get('trial_db_id')
        with self.session() as s:
            trial_model = s.query(Trial).filter_by(id=trial_id).first()
            if trial_model is None:
                self.logger.warning(f'Trial DB id: {trial_id} was not found in DB; cancel update.')
                return
            model_cols = [c.name for c in Trial.__table__.columns]
            for k, v in trial_dict.items():
                if k in model_cols and k not in ['id', 'trial_db_id']:
                    setattr(trial_model, k, v)
            s.commit()

    @commit_func
    def update_block_end_time(self, block_id=None, end_time=None):
        block_id = block_id or self.cache.get(cc.CURRENT_BLOCK_DB_INDEX)
        with self.session() as s:
            block_model = s.query(Block).filter_by(id=block_id).first()
            block_model.end_time = end_time or datetime.now()
            s.commit()
            self.cache.delete(cc.CURRENT_BLOCK_DB_INDEX)

    @commit_func
    def update_experiment_end_time(self, end_time=None):
        end_time = end_time or datetime.now()
        with self.session() as s:
            exp_model = s.query(Experiment).filter_by(id=self.current_experiment_id).first()
            exp_model.end_time = end_time
            s.commit()

    @commit_func
    def commit_temperature(self, temp):
        t = Temperature(time=datetime.now(), value=temp, block_id=self.cache.get(cc.CURRENT_BLOCK_DB_INDEX))
        with self.session() as s:
            s.add(t)
            s.commit()

    def get_temperature(self):
        """return the last temperature value from the last 2 minutes, if none return None"""
        with self.session() as s:
            since = datetime.now() - timedelta(minutes=2)
            temp = s.query(Temperature).filter(Temperature.time > since).order_by(Temperature.time.desc()).first()
            if temp is not None:
                return temp.value

    @commit_func
    def commit_strike(self, strike_dict):
        kwargs = {c.name: strike_dict.get(c.name)
                  for c in Strike.__table__.columns if c.name not in ['id'] and not c.foreign_keys}
        kwargs['block_id'] = strike_dict.get('block_id') or self.cache.get(cc.CURRENT_BLOCK_DB_INDEX)
        kwargs['trial_id'] = strike_dict.get('trial_id')

        with self.session() as s:
            strike = Strike(**kwargs)
            s.add(strike)
            s.commit()

    @commit_func
    def commit_video(self, path, fps, cam_name, start_time, animal_id=None, block_id=None):
        animal_id = animal_id or self.cache.get(cc.CURRENT_ANIMAL_ID)
        vid = Video(path=path, fps=fps, cam_name=cam_name, start_time=start_time,
                    block_id=block_id or self.cache.get(cc.CURRENT_BLOCK_DB_INDEX), animal_id=animal_id)
        with self.session() as s:
            s.add(vid)
            s.commit()
            vid_id = vid.id
        return vid_id

    @commit_func
    def commit_video_frames(self, timestamps: list, video_id: int):
        with self.session() as s:
            video_model = s.query(Video).filter_by(id=video_id).first()
            video_model.frames = {i: ts for i, ts in enumerate(timestamps)}
            video_model.num_frames = len(timestamps)
            video_model.calc_fps = 1 / np.diff(timestamps).mean()
            s.commit()

    @commit_func
    def commit_video_predictions(self, predictor_name: str, data: list, video_id: int, start_time: datetime):
        vid_pred = VideoPrediction(predictor_name=predictor_name, data={i: x for i, x in enumerate(data)},
                                   video_id=video_id, start_time=start_time)
        with self.session() as s:
            s.add(vid_pred)
            s.commit()

    @commit_func
    def commit_pose_estimation(self, cam_name, start_time, x, y, angle, engagement, video_id, model, animal_id=None):
        animal_id = animal_id or self.cache.get(cc.CURRENT_ANIMAL_ID)
        pe = PoseEstimation(cam_name=cam_name, start_time=start_time, x=x, y=y, angle=angle, animal_id=animal_id,
                            engagement=engagement, video_id=video_id, model=model,
                            block_id=self.cache.get(cc.CURRENT_BLOCK_DB_INDEX)
        )
        with self.session() as s:
            s.add(pe)
            s.commit()

    def extract_animal_settings(self, **data):
        kwargs = {}
        for k, v in data.items():
            if k in ANIMAL_SETTINGS_LISTS:
                v = ','.join(v or [])
            kwargs[k] = v
        return kwargs

    @commit_func
    def commit_animal_id(self, **data):
        with self.session() as s:
            kwargs = self.extract_animal_settings(**data)
            animal = Animal(start_time=datetime.now(), **kwargs)
            s.add(animal)
            animal_settings = AnimalSettingsHistory(time=datetime.now(),
                                                    animal_id_key=animal.id, **kwargs)
            s.add(animal_settings)
            s.commit()
            self.cache.set(cc.CURRENT_ANIMAL_ID, data['animal_id'])
            self.cache.set(cc.CURRENT_ANIMAL_ID_DB_INDEX, animal.id)

    @commit_func
    def update_animal_id(self, **kwargs):
        with self.session() as s:
            data = self.extract_animal_settings(**kwargs)
            db_index = self.cache.get(cc.CURRENT_ANIMAL_ID_DB_INDEX)
            if db_index is None:
                return
            animal_model = s.query(Animal).filter_by(id=db_index).first()
            if animal_model is None:
                return
            if 'end_time' not in kwargs:
                animal_settings = AnimalSettingsHistory(time=datetime.now(),
                                                        animal_id_key=animal_model.id, **data)
                s.add(animal_settings)
            for k, v in data.items():
                setattr(animal_model, k, v)
            s.commit()

        if 'end_time' in kwargs:
            self.cache.delete(cc.CURRENT_ANIMAL_ID)
            self.cache.delete(cc.CURRENT_ANIMAL_ID_DB_INDEX)

    def get_animal_settings(self, animal_id):
        with self.session() as s:
            animal = s.query(Animal).filter_by(animal_id=animal_id).first()
            animal_dict = {k: v for k, v in animal.__dict__.items() if not k.startswith('_')}
            for k, v in animal_dict.copy().items():
                if k in ANIMAL_SETTINGS_LISTS:
                    animal_dict[k] = v.split(',')
        return animal_dict

    def get_upcoming_schedules(self):
        with self.session() as s:
            animal_id = self.cache.get(cc.CURRENT_ANIMAL_ID)
            schedules = s.query(Schedule).filter(Schedule.date >= datetime.now(),
                                                 Schedule.animal_id == animal_id).order_by(Schedule.date)
        return schedules

    def commit_schedule(self, date, experiment_name):
        with self.session() as s:
            animal_id = self.cache.get(cc.CURRENT_ANIMAL_ID)
            sch = Schedule(date=date, experiment_name=experiment_name, animal_id=animal_id)
            s.add(sch)
            s.commit()

    def delete_schedule(self, schedule_id):
        with self.session() as s:
            s.query(Schedule).filter_by(id=int(schedule_id)).delete()
            s.commit()

    def commit_reward(self, time):
        with self.session() as s:
            rwd = Reward(time=time,
                         animal_id=self.cache.get(cc.CURRENT_ANIMAL_ID),
                         block_id=self.cache.get(cc.CURRENT_BLOCK_DB_INDEX))
            s.add(rwd)
            s.commit()

    def get_todays_amount_strikes_rewards(self):
        with self.session() as s:
            strikes = s.query(Strike).filter(cast(Strike.time, Date) == date.today()).all()
            rewards = s.query(Reward).filter(cast(Reward.time, Date) == date.today()).all()
        return len(strikes), len(rewards)


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
