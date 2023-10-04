import json
import sys
import time

import pandas as pd
from tqdm.auto import tqdm
from functools import wraps
import numpy as np
from datetime import datetime, timedelta, date
from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, Boolean, create_engine, cast, Date, and_, desc
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
    arena = Column(String)
    bug_types = Column(String)
    reward_bugs = Column(String)
    background_color = Column(String)
    reward_any_touch_prob = Column(Float, default=0)
    exit_hole = Column(String, nullable=True)
    audit = relationship('AnimalSettingsHistory')
    dwh_key = Column(Integer, nullable=True)


class AnimalSettingsHistory(Base):
    __tablename__ = 'animal_settings'

    id = Column(Integer, primary_key=True)
    time = Column(DateTime)
    animal_id = Column(String)
    sex = Column(String)
    arena = Column(String)
    bug_types = Column(String)
    reward_bugs = Column(String)
    background_color = Column(String)
    reward_any_touch_prob = Column(Float, default=0)
    exit_hole = Column(String, nullable=True)
    animal_id_key = Column(Integer, ForeignKey('animals.id'))
    dwh_key = Column(Integer, nullable=True)


class Schedule(Base):
    __tablename__ = 'schedules'

    id = Column(Integer, primary_key=True)
    date = Column(DateTime)
    animal_id = Column(String)
    arena = Column(String)
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
    arena = Column(String)
    blocks = relationship('Block')
    dwh_key = Column(Integer, nullable=True)


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
    dwh_key = Column(Integer, nullable=True)


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
    dwh_key = Column(Integer, nullable=True)


class Temperature(Base):
    __tablename__ = 'temperatures'

    id = Column(Integer, primary_key=True)
    time = Column(DateTime)
    value = Column(Float)
    arena = Column(String)
    sensor = Column(String)
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
    arena = Column(String)
    block_id = Column(Integer, ForeignKey('blocks.id'), nullable=True)
    trial_id = Column(Integer, ForeignKey('trials.id'), nullable=True)
    video_id = Column(Integer, ForeignKey('videos.id'), nullable=True)
    dwh_key = Column(Integer, nullable=True)
    analysis_error = Column(String, nullable=True)


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
    dwh_key = Column(Integer, nullable=True)


class VideoPrediction(Base):
    __tablename__ = 'video_predictions'

    id = Column(Integer, primary_key=True)
    model = Column(String, nullable=True)
    animal_id = Column(String, nullable=True)
    start_time = Column(DateTime)
    arena = Column(String)
    data = Column(JSON)
    video_id = Column(Integer, ForeignKey('videos.id'), nullable=True)
    dwh_key = Column(Integer, nullable=True)


class PoseEstimation(Base):
    __tablename__ = 'pose_estimations'

    id = Column(Integer, primary_key=True)
    cam_name = Column(String)
    start_time = Column(DateTime)
    x = Column(Float)
    y = Column(Float)
    prob = Column(Float, nullable=True)
    bodypart = Column(String, nullable=True)
    model = Column(String, nullable=True)
    animal_id = Column(String, nullable=True)
    angle = Column(Float, nullable=True)
    engagement = Column(Float, nullable=True)
    frame_id = Column(Integer, nullable=True)
    video_id = Column(Integer, ForeignKey('videos.id'), nullable=True)
    block_id = Column(Integer, ForeignKey('blocks.id'), nullable=True)
    dwh_key = Column(Integer, nullable=True)


class Reward(Base):
    __tablename__ = 'rewards'

    id = Column(Integer, primary_key=True)
    time = Column(DateTime)
    animal_id = Column(String)
    arena = Column(String)
    is_manual = Column(Boolean, default=False)
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
                      for c in Experiment.__table__.columns if c.name not in ['id', 'end_time', 'cameras', 'arena', 'dwh_key']}
            kwargs['arena'] = config.ARENA_NAME
            exp_model = Experiment(**kwargs)
            exp_model.cameras = ','.join(list(exp.cameras.keys()))
            s.add(exp_model)
            s.commit()
            self.current_experiment_id = exp_model.id

    @commit_func
    def commit_block(self, blk, is_cache_set=True):
        with self.session() as s:
            kwargs = {c.name: getattr(blk, c.name)
                      for c in Block.__table__.columns if c.name not in ['id', 'end_time', 'dwh_key']
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
                self.cache.set(cc.CURRENT_BLOCK_DB_INDEX, block_id, timeout=blk.overall_block_duration)
        return block_id

    @commit_func
    def commit_trial(self, trial_dict):
        kwargs = {c.name: trial_dict.get(c.name)
                  for c in Trial.__table__.columns if c.name not in ['id', 'dwh_key'] and not c.foreign_keys}
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
                if k in model_cols and k not in ['id', 'trial_db_id', 'dwh_key']:
                    setattr(trial_model, k, v)
            s.commit()

    @commit_func
    def update_block_end_time(self, block_id=None, end_time=None):
        block_id = block_id or self.cache.get(cc.CURRENT_BLOCK_DB_INDEX)
        with self.session() as s:
            block_model = s.query(Block).filter_by(id=block_id).first()
            if block_model is None:
                self.logger.warning(f'No block ID found for end_time update')
                return
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
    def commit_temperature(self, temps):
            with self.session() as s:
                for sensor_name, temp in temps.items():
                    t = Temperature(time=datetime.now(), value=temp, block_id=self.cache.get(cc.CURRENT_BLOCK_DB_INDEX),
                                    arena=config.ARENA_NAME, sensor=sensor_name)
                    s.add(t)
                s.commit()

    def get_temperature(self):
        """return the last temperature value from the last 2 minutes, if none return None"""
        with self.session() as s:
            since = datetime.now() - timedelta(minutes=2)
            temp = s.query(Temperature).filter(and_(Temperature.time > since,
                                               Temperature.arena == config.ARENA_NAME)).order_by(
                Temperature.time.desc()
            ).all()

            res = {}
            for t in temp:
                if t.sensor not in res:
                    res[t.sensor] = t.value
            return res

    @commit_func
    def commit_strike(self, strike_dict):
        kwargs = {c.name: strike_dict.get(c.name)
                  for c in Strike.__table__.columns if c.name not in ['id', 'arena', 'dwh_key'] and not c.foreign_keys}
        kwargs['arena'] = config.ARENA_NAME
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
                    block_id=block_id or self.cache.get(cc.CURRENT_BLOCK_DB_INDEX), animal_id=animal_id,
                    compression_status=0 if not self.cache.get(cc.IS_BLANK_CONTINUOUS_RECORDING) else 1)
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
    def commit_video_predictions(self, model: str, data: pd.DataFrame, video_id: int, start_time: datetime,
                                 animal_id=None, arena=config.ARENA_NAME):
        vid_pred = VideoPrediction(model=model, data=data.to_json(), animal_id=animal_id, arena=arena,
                                   video_id=video_id, start_time=start_time)
        with self.session() as s:
            s.add(vid_pred)
            s.commit()

    @commit_func
    def commit_pose_estimation(self, cam_name, start_time, x, y, angle, engagement, video_id, model,
                               bodypart, prob, frame_id, animal_id=None, block_id=None):
        animal_id = animal_id or self.cache.get(cc.CURRENT_ANIMAL_ID)
        pe = PoseEstimation(cam_name=cam_name, start_time=start_time, x=x, y=y, angle=angle, animal_id=animal_id,
                            engagement=engagement, video_id=video_id, model=model, bodypart=bodypart, prob=prob,
                            frame_id=frame_id, block_id=block_id or self.cache.get(cc.CURRENT_BLOCK_DB_INDEX)
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
            kwargs['arena'] = config.ARENA_NAME
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
            animal = s.query(Animal).filter_by(animal_id=animal_id, arena=config.ARENA_NAME).order_by(
                desc(Animal.start_time)).first()
            if animal is not None:
                animal_dict = {k: v for k, v in animal.__dict__.items() if not k.startswith('_')}
                for k, v in animal_dict.copy().items():
                    if k in ANIMAL_SETTINGS_LISTS:
                        animal_dict[k] = v.split(',')
            else:
                self.logger.error('No Animal was found')
                animal_dict = {}
        return animal_dict

    def get_upcoming_schedules(self):
        with self.session() as s:
            animal_id = self.cache.get(cc.CURRENT_ANIMAL_ID)
            schedules = s.query(Schedule).filter(Schedule.date >= datetime.now(),
                                                 Schedule.animal_id == animal_id,
                                                 Schedule.arena == config.ARENA_NAME).order_by(Schedule.date)
        return schedules

    def commit_multiple_schedules(self, start_date, experiment_name, end_date=None, every=None):
        if not end_date:
            hour, minute = [int(x) for x in config.SCHEDULE_EXPERIMENTS_END_TIME.split(':')]
            end_date = start_date.replace(hour=hour, minute=minute)
        if every:
            curr_date = start_date
            while curr_date < end_date:
                self.commit_schedule(curr_date, experiment_name)
                curr_date += timedelta(minutes=every)
        else:
            self.commit_schedule(start_date, experiment_name)

    def commit_schedule(self, date, experiment_name):
        with self.session() as s:
            animal_id = self.cache.get(cc.CURRENT_ANIMAL_ID)
            sch = Schedule(date=date, experiment_name=experiment_name, animal_id=animal_id, arena=config.ARENA_NAME)
            s.add(sch)
            s.commit()

    def delete_schedule(self, schedule_id):
        with self.session() as s:
            s.query(Schedule).filter_by(id=int(schedule_id)).delete()
            s.commit()

    def commit_reward(self, time, is_manual=False):
        with self.session() as s:
            rwd = Reward(time=time,
                         animal_id=self.cache.get(cc.CURRENT_ANIMAL_ID),
                         block_id=self.cache.get(cc.CURRENT_BLOCK_DB_INDEX),
                         is_manual=is_manual,
                         arena=config.ARENA_NAME)
            s.add(rwd)
            s.commit()

    def get_today_rewards(self, animal_id=None) -> dict:
        with self.session() as s:
            rewards = s.query(Reward).filter(and_(cast(Reward.time, Date) == date.today(),
                                                  Reward.arena == config.ARENA_NAME))
            if animal_id:
                rewards = rewards.filter_by(animal_id=animal_id)
        return {'manual': rewards.filter_by(is_manual=True).count(),
                'auto': rewards.filter_by(is_manual=False).count()}

    def get_today_strikes(self, animal_id=None) -> dict:
        with self.session() as s:
            strks = s.query(Strike).filter(and_(cast(Strike.time, Date) == date.today(),
                                                  Strike.arena == config.ARENA_NAME))
            if animal_id:
                strks = strks.filter_by(animal_id=animal_id)
        return {'hit': strks.filter_by(is_hit=True).count(), 'miss': strks.filter_by(is_hit=False).count()}

    def today_summary(self):
        summary = {}
        with self.session() as s:
            exps = s.query(Experiment).filter(and_(cast(Experiment.start_time, Date) == date.today(),
                                                   Experiment.arena == config.ARENA_NAME)).all()
            for exp in exps:
                summary.setdefault(exp.animal_id, {'total_trials': 0, 'total_strikes': 0, 'blocks': {}})
                for blk in exp.blocks:
                    block_dict = summary[exp.animal_id]['blocks'].setdefault(blk.movement_type, {'hits': 0, 'misses': 0})
                    for tr in blk.trials:
                        summary[exp.animal_id]['total_trials'] += 1
                    for strk in blk.strikes:
                        summary[exp.animal_id]['total_strikes'] += 1
                        if strk.is_hit:
                            block_dict['hits'] += 1
                        else:
                            block_dict['misses'] += 1
        for animal_id, d in summary.items():
            rewards_dict = self.get_today_rewards(animal_id)
            d['total_rewards'] = f'{rewards_dict["auto"]} ({rewards_dict["manual"]})'
        return summary


class DWH:
    commit_models = [Animal, AnimalSettingsHistory, Experiment, Block, Trial, Strike, Video, VideoPrediction]

    def __init__(self):
        self.logger = get_logger('dwh')
        self.local_session = sessionmaker(bind=get_engine())
        self.dwh_session = sessionmaker(bind=create_engine(config.DWH_URL))
        self.keys_table = {}

    def commit(self, n_retries_dwh=3):
        self.logger.info('start DWH commit')
        with self.local_session() as local_s:
            with self.dwh_session() as dwh_s:
                for model in self.commit_models:
                    mappings = []
                    j = 0
                    recs = local_s.query(model).filter(model.dwh_key.is_(None)).all()
                    for rec in tqdm(recs, desc=model.__name__):
                        kwargs = {}
                        for c in model.__table__.columns:
                            if c.name in ['id']:
                                continue
                            value = getattr(rec, c.name)
                            if c.foreign_keys:
                                fk = list(c.foreign_keys)[0]
                                dwh_fk = self.keys_table.get(fk.column.table.name, {}).get(value)
                                if value and not dwh_fk:
                                    # this happened probably due to previously failed runs of DWH commit
                                    dwh_fk = self.get_prev_committed_dwh_fk(local_s, value, fk.column.table)
                                kwargs[c.name] = dwh_fk if value else None
                            else:
                                kwargs[c.name] = value

                        r = model(**kwargs)
                        dwh_s.add(r)
                        dwh_s.commit()
                        self.keys_table.setdefault(model.__table__.name, {})[rec.id] = r.id

                        if model == PoseEstimation:
                            mappings.append({'id': rec.id, 'dwh_key': r.id})
                            j += 1
                            if j % 10000 == 0:
                                local_s.bulk_update_mappings(model, mappings)
                                local_s.flush()
                                local_s.commit()
                                mappings[:] = []
                        else:
                            rec.dwh_key = r.id
                            local_s.commit()

                    if model == PoseEstimation:
                        local_s.bulk_update_mappings(model, mappings)

        self.logger.info('Finished DWH commit')

    def update_model(self, model, columns=()):
        assert isinstance(columns, (list, tuple)), 'columns must be list or tuple'
        with self.local_session() as local_s:
            with self.dwh_session() as dwh_s:
                recs = local_s.query(model).filter(model.dwh_key.is_not(None)).all()
                columns = columns or [c.name for c in model.__table__.columns if c.name in ['id', 'dwh_key'] or c.foreign_keys]
                for rec in tqdm(recs):
                    dwh_rec = dwh_s.query(model).filter_by(id=rec.dwh_key).first()
                    for c in columns:
                        setattr(dwh_rec, c, getattr(rec, c))
                    dwh_s.commit()
                print(f'Finished updating columns={columns} for {model.__name__}; Total rows updated: {len(recs)}')

    @staticmethod
    def get_prev_committed_dwh_fk(s, local_fk, table):
        try:
            return s.query(table).filter_by(id=local_fk).first().dwh_key
        except Exception:
            return


def get_engine():
    return create_engine(config.sqlalchemy_url, pool_size=10, max_overflow=20)


if __name__ == '__main__':
    DWH().commit()
    # DWH().update_model(Strike, ['prediction_distance', 'calc_speed', 'projected_strike_coords', 'projected_leap_coords'])
    sys.exit(0)

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
