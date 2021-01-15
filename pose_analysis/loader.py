import re
import cv2
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import optimize
import matplotlib.pyplot as plt
from functools import cache, cached_property
import sys
sys.path += ['../Arena']
from explore import ExperimentAnalyzer

ROOT_DIR = '/data'
EXPERIMENTS_DIR = ROOT_DIR + '/Pogona_Pursuit/Arena/experiments'
CAMERAS = {
    'realtime': '19506468',
    'right': '19506475',
    'left': '19506455',
    'back': '19506481',
}
SCREEN_BOUNDARIES = {'x': (0, 1850), 'y': (0, 800)}


class Loader:
    def __init__(self, experiment_name=None, trial_id=None, block_id=None, camera=None, video_path=None,
                 experiment_dir=None):
        self.experiment_dir = experiment_dir or EXPERIMENTS_DIR
        if video_path:
            video_path = Path(video_path)
            experiment_name, trial_id, block_id, camera = self.parse_video_path(video_path)
        self.experiment_name = experiment_name
        self.trial_id = trial_id
        self.block_id = block_id
        self.camera = camera
        self.video_path = video_path or self.get_video_path()
        self.validate()
        self.info = self.get_experiment_info()

    def __str__(self):
        return f'{self.experiment_name}/block{self.block_id or 1}/trial{self.trial_id}'

    @cache
    def hits_df(self, hits_only):
        df = pd.read_csv(self.screen_touches_path, index_col=0, parse_dates=['time']).reset_index(drop=True)
        if hits_only:
            df = df.query('is_hit == 1')
        return df

    @cached_property
    def traj_df(self):
        try:
            return pd.read_csv(self.bug_traj_path, index_col=0, parse_dates=['time']).reset_index(drop=True)
        except Exception as exc:
            raise Exception(f'Error loading bug trajectory; {exc}')

    @cached_property
    def traj_time(self):
        return self.traj_df['time'].dt.tz_convert('utc').dt.tz_localize(None)

    @cached_property
    def frames_ts(self) -> pd.Series:
        return pd.to_datetime(pd.read_csv(self.timestamps_path, index_col=0).reset_index(drop=True)['0'])

    def get_frame_at_time(self, t: pd.Timestamp):
        assert isinstance(t, pd.Timestamp)
        if t.tzinfo:
            t = t.tz_convert('utc').tz_localize(None)
        return closest_index(self.frames_ts, t, max_dist=0.080)

    def get_hits_frames(self):
        """return the frame ids for screen strikes"""
        frames = []
        for hit_ts in self.hits_df['time'].dt.tz_convert('utc').dt.tz_localize(None):
            cidx = closest_index(self.frames_ts, hit_ts, max_dist=0.080)
            frames.append(cidx)
        return frames
    
    def get_bug_position_at_time(self, t) -> pd.DataFrame:
        traj_time = self.traj_df['time'].dt.tz_convert('utc').dt.tz_localize(None)
        cidx = closest_index(traj_time, t)
        if cidx is not None:
            return self.traj_df.loc[cidx, ['x', 'y']]

    def bug_data_for_frame(self, frame_id: int) -> pd.DataFrame:
        traj_time = self.traj_df['time'].dt.tz_convert('utc').dt.tz_localize(None)
        frame_time = self.frames_ts[frame_id]
        cidx = closest_index(traj_time, frame_time)
        if cidx is not None:
            return self.traj_df.loc[cidx, :]

    def bug_phases(self):
        def out(a):
            return (self.traj_df[a] < SCREEN_BOUNDARIES[a][0]) | (SCREEN_BOUNDARIES[a][1] < self.traj_df[a])

        in_indices = self.traj_df[~(out('x') | out('y'))].reset_index()['index']
        starts = self.traj_df.loc[in_indices[in_indices.diff() != 1], :]
        ends_indices = in_indices[in_indices[in_indices.diff() > 1].index - 1]
        # add the last in-index as an end_index
        ends_indices = ends_indices.append(pd.Series(in_indices[in_indices.index[-1]], index=[in_indices.index[-1]]))
        ends = self.traj_df.loc[ends_indices, :]
        assert len(starts) == len(ends), 'bad bug_phases analysis, starts != ends'
        return starts, ends

    def validate(self):
        assert self.experiment_path.exists(), 'experiment dir not exist'
        assert self.trial_path.exists(), 'no trial dir'
        assert self.bug_traj_path.exists(), 'no bug trajectory file'
        assert self.screen_touches_path.exists(), 'no screen touches file'
        assert self.video_path.exists(), 'no video file'
        assert self.timestamps_path.exists(), 'no timestamps file'

    @staticmethod
    def parse_video_path(video_path: Path):
        assert video_path.exists(), f'provided video path: {video_path} does not exist'
        try:
            trial_id = int(video_path.parts[-3].split('trial')[1])
            m_block = re.match(r'block(\d+)', video_path.parts[-4])
            if m_block:
                block_id = m_block[1]
                experiment_name = video_path.parts[-5]
            else:
                block_id = None
                experiment_name = video_path.parts[-4]

            camera = None
            for name, serial in CAMERAS.items():
                if name in video_path.name or serial in video_path.name:
                    camera = name
                    break
            if not camera:
                raise Exception('unable to parse camera from video path')
            return experiment_name, trial_id, block_id, camera
        except Exception as exc:
            raise Exception(f'Error parsing video path: {exc}')

    def get_video_path(self) -> Path:
        assert isinstance(self.camera, str), 'no camera name provided or bad type'
        regex = self.camera + r'_\d{8}T\d{6}.(avi|mp4)'
        videos = [v for v in (self.trial_path / 'videos').glob('*') if re.match(regex, v.name)]
        if not videos:
            raise Exception('cannot find video')
        elif len(videos) > 1:
            raise Exception('found more than one video')
        return videos[0]

    def get_experiment_info(self):
        info = ExperimentAnalyzer.get_experiment_info(self.experiment_path / 'experiment.yaml')
        info.update(info.get(f'block{self.block_id or 1}', {}))
        info['trial_id'] = self.trial_id
        return info

    def fit_circle(self):
        assert self.info.get('movement_type') == 'circle', 'Trial must be of circle movement'
        x_center, y_center, _ = fit_circle(self.traj_df.x, -self.traj_df.y)
        assert 800 < x_center < 1600 and -1000 < y_center < -600, 'x,y center are out of range'
        return x_center, y_center

    def is_circle(self):
        return self.info.get('movement_type') == 'circle'

    def get_bug_trajectory_before_strike(self, idx, n_records=20, max_dist=0.050):
        assert idx < len(self.hits_df), f'hit index: {idx} is out of range'
        hit = self.hits_df.loc[idx, :]
        t = hit['time'].tz_convert('utc').tz_localize(None)
        cidx = closest_index(self.traj_time, t, max_dist=max_dist)
        if cidx is None:
            raise Exception(f'unable to find bug trajectory for time: {t};\n'
                            f'bug traj time: {self.traj_time[0]} - {self.traj_time[self.traj_time.index[-1]]}')
        return self.traj_df.loc[cidx - n_records:cidx, ['x', 'y', 'time']]

    def get_rnn_train_set(self, n_records=20, max_hit_dist=1000):
        X = []
        y = []
        infos = []
        traj_time = self.traj_df['time'].dt.tz_convert('utc').dt.tz_localize(None)
        dt = self.traj_df['time'].diff().dt.total_seconds()
        self.traj_df['vx'] = self.traj_df['x'].diff() / dt
        self.traj_df['vy'] = self.traj_df['y'].diff() / dt
        for i, hit in self.hits_df.iterrows():
            t = hit['time'].tz_convert('utc').tz_localize(None)
            cidx = closest_index(traj_time, t)
            if cidx is not None and cidx >= n_records:
                x = self.traj_df.loc[cidx-n_records+1:cidx, ['x', 'y', 'vx', 'vy']].to_numpy()
                if np.isnan(x).any():
                    continue
                hit_dist = distance(hit['x'], hit['y'], hit['bug_x'], hit['bug_y'])
                if max_hit_dist and hist_dist > max_hit_dist:
                    continue
                X.append(x.reshape(1, *x.shape))
                y.append(hit[['x', 'y']].to_numpy().reshape(1, -1))
                info = self.info.copy()
                info['hit_id'] = i
                info['trial_id'] = self.trial_id
                info['is_hit'] = hit['is_hit']
                info['hit_dist'] = hit_dist
                infos.append(info)
                # names.append(f"{self.experiment_name}_trial{self.trial_id}_hit#{i+1}_pogona{self.info['animal_id']}")
        if len(X) > 0:
            return np.concatenate(X), np.concatenate(y), infos
        return None, None, None

    @property
    def animal_id(self):
        return self.info.get('animal_id')

    @property
    def experiment_path(self):
        return self.experiment_dir / Path(self.experiment_name)

    @property
    def trial_path(self):
        if self.block_id:
            return self.experiment_path / f'block{self.block_id}' / f'trial{self.trial_id}'
        return self.experiment_path / f'trial{self.trial_id}'

    @property
    def bug_traj_path(self):
        return self.trial_path / f'bug_trajectory.csv'

    @property
    def screen_touches_path(self):
        return self.trial_path / f'screen_touches.csv'

    @property
    def timestamps_path(self):
        return self.trial_path / 'videos' / 'timestamps' / f'{CAMERAS[self.camera]}.csv'


def distance(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)


def closest_index(series, x, max_dist=0.050):
    diffs = (series - x).abs().dt.total_seconds()
    d = diffs[diffs <= max_dist]
    if len(d) > 0:
        return d.index[d.argmin()]


def get_experiments(*args, **kwargs):
    """Get experiment using explore"""
    df = ExperimentAnalyzer(*args, **kwargs).get_experiments()
    loaders = []
    for experiment, block, trial in df.index:
        try:
            experiment_dir = kwargs.get('experiment_dir') or EXPERIMENTS_DIR
            if not Path(f'{experiment_dir}/{experiment}/block{block}').exists():
                block = None
            ld = Loader(experiment, int(trial), block, 'realtime', experiment_dir=experiment_dir)
            loaders.append(ld)
        except Exception as exc:
            print(f'Error loading {experiment} trial{trial}; {exc}')
            continue
    print(f'num loaders: {len(loaders)}')
    return loaders


def fit_circle(x, y):
    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def f_2(c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = np.mean(x), np.min(y)
    center_2, ier = optimize.leastsq(f_2, center_estimate)

    xc_2, yc_2 = center_2
    Ri_2 = calc_R(*center_2)
    R_2 = Ri_2.mean()
    #     residu_2 = sum((Ri_2 - R_2) ** 2)

    return xc_2, yc_2, R_2