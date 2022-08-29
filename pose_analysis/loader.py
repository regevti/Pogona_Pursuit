import re
import sys
from pathlib import Path
from dateutil import parser
import pandas as pd
import numpy as np
from functools import lru_cache
from pose_analysis.pose_utils import distance, fit_circle, closest_index, pixels2cm
from datetime import timedelta, datetime
from Arena.explore import ExperimentAnalyzer
import pose_analysis.pose_config as config


# SCREEN_BOUNDARIES = {'x': (0, 1918), 'y': (443, 1075)}


class Loader:
    def __init__(self, animal_id=None, day=None, trial_id=None, block_id=None, camera=None, video_path=None,
                 experiment_dir=None, is_validate=True, hits_only=False, label=None):
        if video_path:
            video_path = Path(video_path)
            animal_id, day, trial_id, block_id, camera, experiment_dir = self.parse_video_path(video_path)
        self.experiments_dir = experiment_dir or config.EXPERIMENTS_DIR
        self.animal_id = animal_id
        self.day_dir = day
        self.trial_id = trial_id
        self.block_id = block_id
        self.camera = camera
        self.label = label
        self.info = self.get_experiment_info()
        if is_validate:
            self.video_path = video_path or self.get_video_path()
            self.validate(hits_only)

    def __str__(self):
        return f'{self.day_dir}/block{self.block_id or 1}/trial{self.trial_id}'

    @property
    @lru_cache()
    def hits_df(self):
        if self.screen_touches_path.exists():
            return pd.read_csv(self.screen_touches_path, index_col=0, parse_dates=['time']).reset_index(drop=True)

    @property
    @lru_cache()
    def traj_df(self):
        try:
            return pd.read_csv(self.bug_traj_path, index_col=0, parse_dates=['time']).reset_index(drop=True)
        except Exception as exc:
            raise Exception(f'Error loading bug trajectory; {exc}')

    @property
    @lru_cache()
    def traj_time(self):
        return self.traj_df['time'].dt.tz_convert('utc').dt.tz_localize(None)

    @property
    @lru_cache()
    def frames_ts(self) -> pd.Series:
        return pd.to_datetime(pd.read_csv(self.timestamps_path, index_col=0).reset_index(drop=True)['0'])

    @property
    @lru_cache()
    def calc_speed(self):
        if self.traj_df is None or self.traj_df.empty:
            return
        tf = self.traj_df[['time', 'x', 'y']].diff().iloc[1:, :]
        tf['v'] = np.sqrt((tf.x ** 2) + (tf.y ** 2)) / tf.time.dt.total_seconds()
        return pixels2cm(tf.loc[np.abs(zscore(tf.v)) < 3, 'v'].mean())

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

    def bug_phases(self, mode='bug_on_screen') -> list:
        """get the start and end times on which bug was on screen.
           :return List of tuples with 2 elements (start_time, end_time) which are np.datetime64"""
        def out(a):
            return (self.traj_df[a] < config.SCREEN_BOUNDARIES[a][0]) | (config.SCREEN_BOUNDARIES[a][1] < self.traj_df[a])

        modes = ['bug_on_screen', 'all_show', 'before_show', 'after_show', 'all_trial', 'after_reward_hit']
        assert mode in modes, f'bad mode for position_map: {mode}. options are: {modes}'
        res = []
        try:
            in_indices = self.traj_df[~(out('x') | out('y'))].reset_index()['index']
            if in_indices is not None and len(in_indices) > 0:
                starts = self.traj_time[in_indices[in_indices.diff() != 1]]
                ends_indices = in_indices[in_indices[in_indices.diff() > 1].index - 1]
                # add the last in-index as an end_index
                ends_indices = ends_indices.append(pd.Series(in_indices[in_indices.index[-1]],
                                                             index=[in_indices.index[-1]]))
                ends = self.traj_time[ends_indices]
                assert len(starts) == len(ends), 'bad bug_phases analysis, starts != ends'
                if mode == 'bug_on_screen':
                    return list(zip(starts, ends))
                elif mode == 'all_show':
                    return [(starts.iloc[0], ends.iloc[-1])]
                elif mode == 'before_show' and starts.iloc[0] >= self.frames_ts.iloc[0]:
                    return [(self.frames_ts.iloc[0], starts.iloc[0])]
                elif mode == 'after_show' and self.frames_ts.iloc[-1] >= ends.iloc[-1]:
                    return [(ends.iloc[-1], self.frames_ts.iloc[-1])]
                elif mode == 'all_trial':
                    return [(self.frames_ts.iloc[0], self.frames_ts.iloc[-1])]
                elif mode == 'after_reward_hit':
                    hit_reward_time = self.get_reward_hit_time()
                    if hit_reward_time:
                        return [(hit_reward_time, self.frames_ts.iloc[-1])]

        except Exception as exc:
            print(f'Error in bug_phases: {exc}')

        return res

    def get_reward_hit_time(self):
        if self.hits_df is not None:
            q = self.hits_df.query('is_reward_bug==True')
            return q.time.dt.tz_convert('utc').dt.tz_localize(None).iloc[-1] if not q.empty else None

    def validate(self, hits_only):
        assert self.block_path.exists(), 'experiment dir not exist'
        assert self.trial_path.exists(), 'no trial dir'
        assert self.bug_traj_path.exists(), 'no bug trajectory file'
        assert self.video_path.exists(), 'no video file'
        assert self.timestamps_path.exists(), 'no timestamps file'
        if hits_only:
            assert self.screen_touches_path.exists(), 'no screen touches file'

    @staticmethod
    def parse_video_path(video_path: Path):
        assert video_path.exists(), f'provided video path: {video_path} does not exist'
        try:
            trial_id = int(video_path.parts[-3].split('trial')[1])
            block_id = re.match(r'block(\d+)', video_path.parts[-4])[1]
            day_dir = video_path.parts[-5]
            animal_id = video_path.parts[-6]
            experiment_dir = Path(*video_path.parts[:-6]).as_posix()

            camera = None
            for name, serial in config.CAMERAS.items():
                if name in video_path.name or serial in video_path.name:
                    camera = name
                    break
            if not camera:
                raise Exception('unable to parse camera from video path')
            return animal_id, day_dir, trial_id, block_id, camera, experiment_dir
        except Exception as exc:
            raise Exception(f'Error parsing video path: {exc}')

    def get_video_path(self) -> Path:
        assert isinstance(self.camera, str), 'no camera name provided or bad type'
        regex = self.camera + r'_\d{8}T\d{6}.(avi|mp4)'
        vids_path = self.trial_path / 'videos'
        videos = [v for v in vids_path.glob('*') if re.match(regex, v.name)]
        if not videos:
            raise Exception(f'cannot find videos in {vids_path}')
        elif len(videos) > 1:
            raise Exception(f'found more than one video in {vids_path}')
        return videos[0]

    def get_experiment_info(self):
        info = ExperimentAnalyzer.get_block_info(self.block_path / 'info.yaml')
        # info.update(info.get(f'block{self.block_id or 1}', {}))
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
                if max_hit_dist and hit_dist > max_hit_dist:
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

    def is_v1(self):
        return self.version == '1'

    @property
    def day(self):
        return self.day_dir

    @property
    def block_path(self):
        return self.experiments_dir / Path(self.animal_id) / self.day_dir / f'block{self.block_id}'

    @property
    def trial_path(self):
        if self.is_v1():
            return self.block_path / f'trial{self.trial_id}'
        return self.block_path

    @property
    def bug_traj_path(self):
        return self.trial_path / f'bug_trajectory.csv'

    @property
    def screen_touches_path(self):
        return self.trial_path / f'screen_touches.csv'

    @property
    def timestamps_path(self):
        return self.trial_path / 'videos' / 'timestamps' / f'{config.CAMERAS[self.camera]}.csv'

    @property
    def version(self):
        version = self.info.get('version', '2.0')
        return re.match(r'(\d).\d+', version)[1]


def get_experiments(*args, is_validate=True, **kwargs):
    """Get experiment using explore"""
    df = ExperimentAnalyzer(*args, **kwargs).get_experiments()
    loaders = []
    for animal_id, day, block, trial in df.index:
        day_dir = datetime.strptime(day, '%d.%m.%y').strftime('%Y%m%d')
        try:
            experiments_dir = kwargs.get('experiment_dir') or config.EXPERIMENTS_DIR
            label = df.loc[(animal_id, day, block, trial), 'bad_label']
            ld = Loader(animal_id, day_dir, int(trial), block, 'realtime',
                        experiment_dir=experiments_dir, is_validate=is_validate, label=label)
            loaders.append(ld)
        except Exception as exc:
            print(f'Error loading {animal_id}/{day_dir}/block{block}/trial{trial}; {exc}')
            continue
    print(f'num loaders: {len(loaders)}')
    return loaders
