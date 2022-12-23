import pickle
import pandas as pd
from db_models import ORM, Video, Experiment, Trial, PoseEstimation, Strike, \
    Block, Temperature, VideoPrediction
from tqdm.auto import tqdm
from datetime import datetime, timedelta
import random
from pathlib import Path
import numpy as np

screen_width, screen_height = 1080, 700


class A:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


PRED_DIR = '../output/predictions'


def download_predictions(**filters):
    orm = ORM()
    with orm.session() as s:
        for vp in tqdm(s.query(VideoPrediction).filter_by(**filters).all()):
            vid_path = Path(s.query(Video).filter_by(id=vp.video_id).first().path)
            with open(f'{PRED_DIR}/{vid_path.stem}.pickle', 'wb') as f:
                pickle.dump({'data': vp.data, 'predictor_name': vp.predictor_name,
                             'start_time': vp.start_time,
                             'video_path': str(vid_path)}, f)

# predictor_name: str, data: list, video_id: int, start_time
def upload_predictions():
    orm = ORM()
    with orm.session() as s:
        for p in Path(PRED_DIR).glob('*.pickle'):
            with p.open('rb') as f:
                d = pickle.load(f)
                vid = s.query(Video).filter_by(path=d.pop('video_path')).first()
                orm.commit_video_predictions(d['predictor_name'], d['data'].values(), vid.id, d['start_time'])


def main(animal_id):
    orm = ORM()
    with orm.session() as s:
        for e in s.query(Experiment).filter_by(animal_id=animal_id).all():
            for blk in e.blocks:
                for strk in blk.strikes:
                    s.delete(strk)
                for tr in blk.trials:
                    s.delete(tr)
                for v in blk.videos:
                    s.delete(v)
                s.query(Temperature).filter_by(block_id=blk.id).delete()
                s.query(PoseEstimation).filter_by(block_id=blk.id).delete()
                for v in s.query(Video).filter_by(block_id=blk.id).all():
                    s.query(VideoPrediction).filter_by(video_id=v.id).delete()
                    s.delete(v)
                s.delete(blk)
            s.delete(e)
        s.commit()


if __name__ == '__main__':
    # main(animal_id='PV87')
    # download_predictions(predictor_name='pogona_head_local')
    upload_predictions()

    # experiments = {
    #     16: 'random',
    #     17: 'random',
    #     18: 'random',
    #     19: 'circle',
    #     20: 'circle',
    #     21: 'circle',
    #     22: None,
    #     23: 'horizontal+noise',
    #     24: 'horizontal+noise',
    #     25: 'horizontal+noise',
    #     26: 'jump',
    #     27: 'jump',
    # }
    # for day, exp_name in experiments.items():
    #     if not exp_name:
    #         continue
    #     start_time = datetime(year=2022, month=10, day=22, hour=9, minute=0)
    #     orm.commit_experiment(A(name=exp_name, start_time=start_time, end_time=None,
    #                             animal_id=animal_id, cameras={'top':1,'front':1,'back':1}, num_blocks=4,
    #                             extra_time_recording=30, time_between_blocks=10, experiment_path=''))
    #
    #     for i in range(6):
    #         num_trials = 5
    #         block_start = start_time + timedelta(hours=i+1)
    #         orm.commit_block(A(start_time=block_start, end_time=None,
    #                            block_id=i + 1, num_trials=num_trials, trial_duration=40, iti=30, block_type=exp_name,
    #                            bug_types='cockroach,red_beetle', bug_speed=4, bug_size=100,
    #                            is_default_bug_size=True, exit_hole='bottomLeft', reward_type='always',
    #                            reward_bugs='cockroach,red_beetle', reward_any_touch_prob=1, media_url='',
    #                            movement_type=exp_name, is_anticlockwise=True, target_drift='left',
    #                            bug_height=20, time_between_bugs=2000, background_color=None))
    #
    #         for trial_id in range(1, num_trials+1):
    #             trial_db_index = orm.commit_trial({
    #                 'in_block_trial_id': trial_id,
    #                 'start_time': block_start + timedelta(seconds=40*trial_id),
    #                 'bug_trajectory': {}
    #             })
    #             num_strikes = random.randint(0, 3)
    #             for j in range(num_strikes):
    #                 is_last = j == num_strikes - 1
    #                 if exp_name in ['random', 'jump']:
    #                     bug_x, bug_y = random.random() * screen_width, random.random() * screen_height
    #                 elif exp_name == 'circle':
    #                     r = screen_width / 3
    #                     r0 = (screen_width/2, 0)
    #                     bug_x = r0[0] + r * np.cos(random.random() * np.pi)
    #                     bug_y = r0[1] + r * np.sin(random.random() * np.pi)
    #                 elif exp_name == 'horizontal+noise':
    #                     bug_y = 200 + random.random() * 2
    #                     bug_x = random.random() * screen_width
    #                 x, y = random.random() * screen_width, random.random() * screen_height
    #                 while dist(x, y, bug_x, bug_y) > (2 if is_last else 10):
    #                     x, y = random.random() * screen_width, random.random() * screen_height
    #                 orm.commit_strike({'time': block_start + timedelta(seconds=40*trial_id), 'is_hit': is_last,
    #                                    'is_reward_bug': True, 'is_climbing': False,
    #                                    'x': x, 'y': y, 'bug_x': bug_x, 'bug_y': bug_y,
    #                                    'bug_type': '', 'bug_size': 20, 'in_block_trial_id': trial_id,
    #                                    'trial_id': trial_db_index})