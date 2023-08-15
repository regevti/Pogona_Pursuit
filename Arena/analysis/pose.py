import datetime
import math
import yaml
import cv2
import traceback
import importlib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from scipy.spatial import distance
from scipy.signal import savgol_filter
from scipy.stats import ttest_ind
from multiprocessing.pool import ThreadPool
import os
if Path('.').resolve().name != 'Arena':
    os.chdir('..')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import config
from calibration import CharucoEstimator
from loggers import get_logger
from utils import run_in_thread, Kalman
from sqlalchemy import cast, Date
from db_models import ORM, Experiment, Block, Video, VideoPrediction, Strike, PoseEstimation
from image_handlers.video_writers import OpenCVWriter
from analysis.pose_utils import put_text, flatten


MIN_DISTANCE = 5  # cm
COMMIT_INTERVAL = 2  # seconds
VELOCITY_SAMPLING_DURATION = 2  # seconds
COMMIT_DB_BODYPARTS = ['nose', 'left_ear', 'right_ear']


class MissingFile(Exception):
    """"""


class NoFramesVideo(Exception):
    """"""


class ArenaPose:
    def __init__(self, cam_name, predictor, is_use_db=True, orm=None, model_path=None):
        self.cam_name = cam_name
        self.predictor = predictor
        self.model_path = model_path
        self.is_use_db = is_use_db
        self.load_predictor()
        self.last_commit = None
        self.caliber = None
        self.orm = orm if orm is not None else ORM()
        self.commit_bodypart = 'head'
        self.kinematic_cols = ['x', 'y', 'vx', 'vy', 'ax', 'ay']
        self.time_col = ('time', '')
        self.kalman = None
        self.predictions = []
        self.current_position = (None, None)
        self.current_velocity = None
        self.is_initialized = False
        self.example_writer = None
        # self.screen_coords = get_screen_coords('pogona_pursuit2')

    def init(self, img, caliber_only=False):
        if not caliber_only:
            self.predictor.init(img)
        self.caliber = CharucoEstimator(self.cam_name, is_debug=False)
        self.caliber.init(img)
        self.is_initialized = True
        if not self.caliber.is_on:
            raise Exception('Could not initiate caliber; closing ArenaPose')

    def init_from_video(self, video_path: [str, Path], caliber_only=False):
        if isinstance(video_path, Path):
            video_path = video_path.as_posix()
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if frame is None:
            raise Exception('Video has 0 frames')
        self.init(frame, caliber_only=caliber_only)
        cap.release()

    def load(self, video_path=None, video_db_id=None, only_load=False, prefix=''):
        if self.is_use_db:
            assert video_db_id, 'must provide video_db_id if is_use_db=True'
            return self._load_from_db(video_db_id)
        else:
            assert video_path, 'must provide video_path if is_use_db=False'
            return self._load_from_local_files(video_path, only_load, prefix)

    def _load_from_db(self, video_db_id):
        with self.orm.session() as s:
            pose_estimations = s.query(PoseEstimation).filter_by(video_id=video_db_id,
                                                                 model=self.predictor.model_name).all()
        dfs = {}
        for pe in pose_estimations:
            row = {('time', ''): datetime.datetime.timestamp(pe.start_time), (pe.bodypart, 'x'): pe.x, (pe.bodypart, 'y'): pe.y,
                   (pe.bodypart, 'prob'): pe.prob}
            if pe.bodypart == 'nose':
                row.update({('angle', ''): pe.angle})
            dfs.setdefault(pe.bodypart, []).append(row)

        df = pd.DataFrame(dfs['nose'])
        for bp in COMMIT_DB_BODYPARTS[1:]:
            df = df.merge(pd.DataFrame(dfs[bp]), on=[('time', '')], how='outer')

        df.columns = pd.MultiIndex.from_tuples(df.columns)
        df = df.sort_values(by='time')
        return df

    def _load_from_local_files(self, video_path: Path, only_load=False, prefix=''):
        if isinstance(video_path, str):
            video_path = Path(video_path)
        if not self.is_initialized:
            self.init_from_video(video_path, caliber_only=True)
        cache_path = self.get_predicted_cache_path(video_path)
        if cache_path.exists():
            pose_df = pd.read_parquet(cache_path)
        else:
            if not only_load:
                pose_df = self.predict_video(video_path=video_path, prefix=prefix)
            else:
                raise MissingFile(f'Pose cache file does not exist')
        return pose_df

    def test_loaders(self, db_video_id):
        with self.orm.session() as s:
            video_path = s.query(Video).filter_by(id=db_video_id).first().path
            assert video_path and Path(video_path).exists()

        pose_df_local = self._load_from_local_files(video_path, only_load=True)
        pose_df_db = self._load_from_db(db_video_id)
        # pose_df_db[('time', '')] = pose_df_db['time'].map(lambda x: datetime.datetime.timestamp(x))

        fig, axes = plt.subplots(1, 3, figsize=(25, 8))
        for i, bp in enumerate(COMMIT_DB_BODYPARTS):
            axes[i].plot(pose_df_local['time']-pose_df_local['time'].iloc[0], pose_df_local[bp]['y'], '-o', label='local')
            axes[i].plot(pose_df_db['time']-pose_df_db['time'].iloc[0], pose_df_db[bp]['y'], '-o', label='db')
            axes[i].legend()
        fig.tight_layout()
        plt.show()

    def start_new_session(self, fps):
        self.kalman = Kalman(dt=1/fps)
        self.predictions = []

    def load_predictor(self):
        if isinstance(self.predictor, str):
            prd_module, prd_class = config.arena_modules['predictors'][self.predictor]
            prd_module = importlib.import_module(prd_module)
            self.predictor = getattr(prd_module, prd_class)(self.cam_name, self.model_path)

    def predict_video(self, db_video_id=None, video_path=None, is_save_cache=True, is_create_example_video=False,
                      prefix=''):
        """
        predict pose for a given video
        @param db_video_id: The DB index of the video in the videos table
        @param video_path: The path of the video
        @param is_save_cache: save predicted dataframe as parquet file
        @param is_create_example_video: create annotated video with predictions
        @param prefix: to be displayed before the tqdm desc
        @return:
        """
        db_video_id, video_path = self.check_video_inputs(db_video_id, video_path)
        frames_times = self.load_frames_times(db_video_id, video_path)

        pose_df = []
        cap = cv2.VideoCapture(video_path)
        n_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(frames_times))
        if n_frames == 0:
            self.tag_error_video(video_path, 'video has 0 frames')
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.start_new_session(fps)
        for frame_id in tqdm(range(n_frames), desc=f'{prefix}{Path(video_path).stem}'):
            ret, frame = cap.read()
            if not self.is_initialized:
                self.init(frame)

            timestamp = frames_times.loc[frame_id, 'time'].timestamp()
            pred_row = self.predictor.predict(frame, frame_id)
            pred_row = self.analyze_frame(timestamp, pred_row, db_video_id)
            if is_create_example_video:
                self.write_to_example_video(frame, frame_id, pred_row, fps, video_path)
            pose_df.append(pred_row)
        cap.release()

        if not pose_df:
            return
        pose_df = pd.concat(pose_df)
        if is_save_cache:
            self.save_predicted_video(pose_df, video_path)
        self.close_example_writer()
        return pose_df

    def tag_error_video(self, vid_path, msg):
        with self.get_predicted_cache_path(vid_path).with_suffix('.txt').open('w') as f:
            f.write(msg)

    def predict_frame(self, frame, frame_id) -> pd.DataFrame:
        raise NotImplemented('method predict_frame is not implemented')

    def analyze_frame(self, timestamp: float, pred_row: pd.DataFrame, db_video_id=None):
        """
        Convert pose in pixels to real-world coordinates using the calibrator, and smooth using Kalman.
        In addition, if is_commit is enabled this method would commit to DB the predictions for body parts in
        self.commit_bodypart
        """
        pred_row[self.time_col] = timestamp
        for col in self.kinematic_cols:
            pred_row[(col, '')] = np.nan

        for bodypart in self.predictor.bodyparts:
            if self.is_ready_to_commit(timestamp) and bodypart == self.commit_bodypart:
                # if predictions stack has reached the COMMIT_INTERVAL, and it's the commit_bodypart,
                # then calculate aggregated metrics (e.g, velocity).
                self.aggregate_to_commit(self.predictions.copy(), db_video_id, timestamp)

            cam_x, cam_y = pred_row[(bodypart, 'cam_x')].iloc[0], pred_row[(bodypart, 'cam_y')].iloc[0]
            x, y = np.nan, np.nan
            if not np.isnan(cam_x) and not np.isnan(cam_y):
                x, y = self.caliber.get_location(cam_x, cam_y)

            if bodypart == self.commit_bodypart:
                if not self.kalman.is_initiated:
                    self.kalman.init(x, y)
                x, y, vx, vy, ax, ay = self.kalman.get_filtered(x, y)
                for col in self.kinematic_cols:
                    pred_row.loc[pred_row.index[0], (col, '')] = locals()[col]
                if not np.isnan(x) or not np.isnan(y):
                    self.predictions.append((timestamp, x, y))
            else:
                pred_row.loc[pred_row.index[0], (bodypart, 'x')] = x
                pred_row.loc[pred_row.index[0], (bodypart, 'y')] = y

        pred_row = self.after_analysis_actions(pred_row)
        return pred_row

    def check_video_inputs(self, db_video_id, video_path):
        if self.is_use_db and not db_video_id and video_path:
            db_video_id = self.get_video_db_id(video_path)
        elif not video_path and db_video_id:
            if not self.is_use_db:
                raise Exception('must use DB to get video path but is_use_db=False')
            video_path = self.get_video_path(db_video_id)
        video_path = Path(video_path).as_posix()
        return db_video_id, video_path

    def aggregate_to_commit(self, predictions, db_video_id, timestamp):
        predictions = np.array(predictions)
        x, y = [round(z) for z in predictions[:, 1:3].mean(axis=0)]
        if self.is_use_db and self.is_moved(x, y):
            self.commit_to_db(timestamp, x, y, db_video_id)
        self.predictions = []

    @run_in_thread
    def commit_to_db(self, timestamp, x, y, db_video_id):
        start_time = datetime.datetime.fromtimestamp(timestamp)
        self.orm.commit_pose_estimation(self.cam_name, start_time, x, y, None, None,
                                        db_video_id, model='deeplabcut_v1')
        self.last_commit = (timestamp, x, y)

    def after_analysis_actions(self, pred_row):
        return pred_row

    def load_frames_times(self, db_video_id: int, video_path: str) -> pd.DataFrame:
        if self.is_use_db:
            with self.orm.session() as s:
                vid = s.query(Video).filter_by(id=db_video_id).first()
                if vid.frames is None:
                    raise MissingFile(f'unable to find frames_timestamps in DB for video_id: {db_video_id}')
                frames_ts = pd.DataFrame(vid.frames.items(), columns=['frame_id', 'time']).set_index('frame_id')
        else:
            frames_output_dir = Path(video_path).parent / config.frames_timestamps_dir
            csv_path = frames_output_dir / Path(video_path).with_suffix('.csv').name
            if not csv_path.exists():
                raise MissingFile(f'unable to find frames_timestamps in {csv_path}')
            frames_ts = pd.read_csv(csv_path, names=['frame_id', 'time'], header=0).set_index('frame_id')

        frames_ts['time'] = pd.to_datetime(frames_ts.time, unit='s', utc=True).dt.tz_convert(
            'Asia/Jerusalem').dt.tz_localize(None)
        frames_ts.index = frames_ts.index.astype(int)
        return frames_ts

    def get_video_path(self, db_video_id: int) -> str:
        with self.orm.session() as s:
            vid = s.query(Video).filter_by(id=db_video_id).first()
            if vid is None:
                raise Exception(f'unable to find video_id: {db_video_id}')
            return vid.path

    def get_video_db_id(self, video_path: Path):
        with self.orm.session() as s:
            vid = s.query(Video).filter(Video.path.contains(video_path.stem)).first()
            if vid is None:
                raise Exception(f'unable to find video path: {video_path}')
            return vid.id

    def load_predicted_video(self, video_path):
        cache_path = self.get_predicted_cache_path(video_path)
        if not cache_path.exists():
            raise MissingFile(f'No prediction cache found under: {cache_path}')
        pose_df = pd.read_parquet(cache_path)
        return pose_df

    def save_predicted_video(self, pose_df, video_path):
        cache_path = self.get_predicted_cache_path(video_path)
        pose_df.to_parquet(cache_path)

    def write_to_example_video(self, frame, frame_id, pred_row, fps, video_path):
        if self.example_writer is None:
            example_path = self.get_predicted_cache_path(video_path).with_suffix('.avi').as_posix()
            self.example_writer = OpenCVWriter(frame, fps, is_color=True, full_path=example_path)

        frame = self.predictor.plot_predictions(frame, frame_id, pred_row)
        x, y = 40, 200
        for col in self.kinematic_cols:
            frame = put_text(f'{col}={pred_row[col].iloc[0]:.1f}', frame, x, y)
            y += 30
        if 'angle' in pred_row.columns:
            frame = put_text(f'angle={math.degrees(pred_row["angle"].iloc[0]):.1f}', frame, x, y)
        self.example_writer.write(frame)

    def close_example_writer(self):
        if self.example_writer is not None:
            self.example_writer.close()
            self.example_writer = None

    def get_predicted_cache_path(self, video_path) -> Path:
        preds_dir = Path(video_path).parent / 'predictions'
        preds_dir.mkdir(exist_ok=True)
        vid_name = Path(video_path).with_suffix('.parquet').name
        return preds_dir / f'{self.predictor.model_name}__{vid_name}'

    def is_moved(self, x, y):
        return not self.last_commit or distance.euclidean(self.last_commit[1:], (x, y)) < MIN_DISTANCE

    def is_ready_to_commit(self, timestamp):
        return self.predictions and (timestamp - self.predictions[0][0]) > COMMIT_INTERVAL


class DLCArenaPose(ArenaPose):
    def __init__(self, cam_name, is_use_db=True, orm=None, **kwargs):
        super().__init__(cam_name, 'deeplabcut', is_use_db, orm, **kwargs)
        self.commit_bodypart = 'mid_ears'
        self.pose_df = pd.DataFrame()
        self.angle_col = ('angle', '')

    def after_analysis_actions(self, pred_row):
        angle = self.calc_head_angle(pred_row.iloc[0])
        pred_row.loc[pred_row.index[0], self.angle_col] = angle
        return pred_row

    @staticmethod
    def calc_head_angle(row):
        x_nose, y_nose = row.nose.x, row.nose.y
        x_ears = (row.right_ear.x + row.left_ear.x) / 2
        y_ears = (row.right_ear.y + row.left_ear.y) / 2
        dy = y_ears - y_nose
        dx = x_ears - x_nose
        if dx != 0.0:
            theta = np.arctan(abs(dy) / abs(dx))
        else:
            theta = np.pi / 2
        if dx > 0:  # looking south
            theta = np.pi - theta
        if dy < 0:  # looking opposite the screen
            theta = -1 * theta
        if theta < 0:
            theta = 2 * np.pi + theta
        return theta

    @property
    def body_parts(self):
        return [b for b in self.pose_df.columns.get_level_values(0).unique()
                if b and isinstance(self.pose_df[b], pd.DataFrame)]


class SpatialAnalyzer:
    def __init__(self, animal_id=None, day=None, cam_name='front', bodypart='mid_ears', split_by=None, orm=None,
                 is_use_db=False, **block_kwargs):
        """
        Spatial analysis and visualization
        @param animal_id:
        @param day: limit results to a certain day
        @param movement_type:
        @param cam_name:
        @param bodypart: The body part from which the pose coordinates will be taken
        """
        self.animal_id = animal_id
        self.day = day
        self.cam_name = cam_name
        self.bodypart = bodypart
        assert split_by is None or isinstance(split_by, list), 'split_by must be a list of strings'
        self.split_by = split_by
        self.block_kwargs = block_kwargs
        self.orm = orm if orm is not None else ORM()
        self.dlc = DLCArenaPose('front', is_use_db=is_use_db, orm=self.orm)
        self.coords = {
            'arena': np.array([(-3, -2), (42, 78)]),
            'arena_close': np.array([(-3, -2), (42, 15)]),
            'screen': np.array([(-1, -3), (39, -1)])
        }

    def get_pose(self):
        res = {}
        for group_name, vids in self.get_videos_paths().items():
            for video_path in vids:
                pose_df = self._load_pose(video_path)
                res.setdefault(group_name, []).append(pose_df)

        for group_name in res.copy().keys():
            res[group_name] = pd.concat(res[group_name])
        return res

    def _load_pose(self, video_path):
        pose_df = self.dlc.load(video_path, only_load=True)
        if self.bodypart == 'mid_ears':
            for c in ['x', 'y']:
                pose_df[('mid_ears', c)] = pose_df[[('right_ear', c), ('left_ear', c)]].mean(axis=1)
            pose_df[('mid_ears', 'prob')] = pose_df[[('right_ear', 'prob'), ('left_ear', 'prob')]].min(axis=1)
        return pd.concat([pd.to_datetime(pose_df['time'], unit='s'), pose_df[self.bodypart]], axis=1)

    def get_videos_paths(self, is_add_block_video_id=False) -> dict:
        """return list of lists of groups of video paths that are split using 'split_by'"""
        video_paths = {}
        with self.orm.session() as s:
            exps = s.query(Experiment)
            if self.animal_id:
                exps = exps.filter_by(animal_id=self.animal_id)
            else:
                exps = exps.filter(Experiment.animal_id.not_in(['test']))
            if self.day:
                exps = exps.filter(cast(Experiment.start_time, Date) == self.day)
            for exp in exps.all():
                for blk in exp.blocks:
                    if self.block_kwargs and any(getattr(blk, k) != v for k, v in self.block_kwargs.items()):
                        continue

                    if self.split_by:
                        group_name = ','.join([f"{c}={getattr(blk, c, None)}" for c in self.split_by])
                    else:
                        group_name = ''

                    for vid in blk.videos:
                        if self.cam_name and vid.cam_name != self.cam_name:
                            continue
                        if not is_add_block_video_id:
                            v = vid.path
                        else:
                            v = (vid.path, blk.id, vid.id)
                        video_paths.setdefault(group_name, []).append(v)

        return video_paths

    def drop_out_of_arena_coords(self, df):
        xmin, xmax = self.coords['arena'][:, 0].flatten().tolist()
        idx = df[(df.y < xmin) | (df.y > xmax)].index
        return df.drop(idx)

    def get_out_of_experiment_pose(self):
        groups_pose = {}
        for group_name, vids in self.get_videos_paths().items():
            day_paths = list(set([Path(*Path(v).parts[:-3]) for v in vids]))
            pose_ = []
            for day_p in day_paths:
                tracking_dir = day_p / 'tracking' / 'predictions'
                if not tracking_dir.exists():
                    continue

                for p in tracking_dir.rglob('*.csv'):
                    df = pd.read_csv(p, index_col=0)
                    df = df[~df.x.isna()]
                    pose_.extend(df[['x', 'y']].to_records(index=False).tolist())
            groups_pose[group_name] = pose_
        return groups_pose

    def plot_out_of_experiment_pose(self, axes=None):
        groups_pose = self.get_out_of_experiment_pose()
        axes = self.get_axes(4, len(groups_pose), axes, is_cbar=False)
        for i, (group_name, pose_list) in enumerate(groups_pose.items()):
            df = pd.DataFrame(pose_list, columns=['x', 'y'])
            sns.histplot(data=df, x='x', y='y', ax=axes[i], bins=(30, 30), cmap='Greens', stat='probability')
            axes[i].set_xlim([0, 50])
        plt.show()

    def plot_spatial(self, pose_dict=None, cols=4, axes=None):
        if pose_dict is None:
            pose_dict = self.get_pose()

        axes_ = self.get_axes(cols, len(pose_dict), axes=axes)
        cbar_ax = None
        for i, (group_name, pose_df) in enumerate(pose_dict.items()):
            if i == len(pose_dict) - 1:
                cbar_ax = axes_[-1]
            self.plot_hist2d(pose_df, axes_[i], cbar_ax=cbar_ax)
            self.plot_arena(axes_[i])
            if len(self.split_by) == 1 and self.split_by[0] == 'exit_hole':
                group_name = 'Left --> Right' if 'bottomRight' in group_name else 'Right --> Left'
            axes_[i].set_title(group_name)
        if axes is None:
            plt.tight_layout()
            plt.show()

    @staticmethod
    def plot_hist2d(df, ax, cbar_ax=None):
        df_ = df.query('0 <= x <= 40 and y<10')
        sns.histplot(data=df_, x='x', y='y', ax=ax,
                     bins=(30, 10), cmap='Greens', stat='probability',
                     cbar=cbar_ax is not None, cbar_kws=dict(shrink=.75, label='Probability'), cbar_ax=cbar_ax)
        ax.set_yticks([0, 10])
        ax.set_xticks([0, 20, 40])
        ax.set_ylabel(None)

        inner_ax = inset_axes(ax, width="70%", height="60%", loc='upper right', borderpad=1)
        sns.kdeplot(data=df_, x='x', ax=inner_ax, clip=[0, 40])
        inner_ax.axvline(20, linestyle='--', color='tab:orange')
        inner_ax.set_xticks([0, 20, 40])
        inner_ax.set_ylim([0, 0.15])
        inner_ax.set_yticks([0, 0.05, 0.1])
        inner_ax.set_ylabel(None)

    def plot_arena(self, ax, is_close_to_screen_only=False):
        for name, c in self.coords.items():
            rect = patches.Rectangle(c[0, :], *(c[1, :] - c[0, :]).tolist(), linewidth=1, edgecolor='k',
                                     facecolor='k' if name == 'screen' else 'none')
            ax.add_patch(rect)
        ax.invert_xaxis()
        ax.set_xlim(self.coords['arena' if not is_close_to_screen_only else 'arena_close'][:, 0])
        ax.set_ylim(self.coords['arena' if not is_close_to_screen_only else 'arena_close'][:, 1])

    def plot_trajectories(self, cols=4, axes=None, only_to_screen=False):
        pose_dict = self.get_pose()
        axes = self.get_axes(cols, len(pose_dict), axes, is_cbar=False)

        for i, (group_name, pose_df) in enumerate(pose_dict.items()):
            trajs = self.cluster_trajectories(pose_df, only_to_screen=only_to_screen)
            for j, traj in trajs.items():
                traj = np.array(traj)
                # remove NaNs
                traj = traj[~np.isnan(traj).any(axis=1), :]
                total_distance = np.sum(np.sqrt(np.sum(np.diff(traj, axis=0) ** 2, axis=1)))
                if total_distance < 5:
                    continue

                # axes[i].plot(traj[:, 0], traj[:, 1], label=f'traj{j}')
                axes[i].plot([traj[0, 0], traj[-1, 0]], [traj[0, 1], traj[-1, 1]], label=f'traj{j}')
                axes[i].scatter(traj[-1, 0], traj[-1, 1], marker='*')
                axes[i].set_title(group_name)

            x_values = np.concatenate([t.x.values for t in trajs.values()])
            inner_ax = inset_axes(axes[i], width="70%", height="40%", loc='upper left', borderpad=1)
            sns.kdeplot(x=x_values, ax=inner_ax, clip=[0, 40])
            inner_ax.axvline(20, linestyle='--', color='tab:orange')
            inner_ax.set_xticks([0, 20, 40])
            inner_ax.set_ylim([0, 0.15])
            inner_ax.set_yticks([0, 0.05, 0.1])
            inner_ax.set_ylabel(None)



            self.plot_arena(axes[i], is_close_to_screen_only=True)
            axes[i].legend()

        plt.show()

    def play_trajectories(self, video_path: str, only_to_screen=False):
        pose_df = self._load_pose(video_path)
        cap = cv2.VideoCapture(video_path)
        trajs = self.cluster_trajectories(pose_df, only_to_screen=only_to_screen)

        for start_frame, traj in trajs.items():
            total_distance = self.calc_traj_distance(traj)
            self.play_segment(cap, start_frame, len(traj), f'Traj{start_frame} ({total_distance:.1f})')

        cap.release()

    @staticmethod
    def play_segment(cap, start_frame, n_frames, frames_text=None):
        assert frames_text is None or isinstance(frames_text, str) or len(frames_text) == n_frames, \
            'frames_text must be a string or a list in the length of n_frames'
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for i in range(n_frames):
            ret, frame = cap.read()
            frame = cv2.resize(frame, None, None, fx=0.5, fy=0.5)
            if frames_text:
                put_text(frames_text[i] if isinstance(frames_text, list) else frames_text, frame, 30, 20)
            cv2.imshow('Pogona Pose', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

    @staticmethod
    def get_cross_limit(cross_id: int, s: pd.Series, grace=5, dist_threshold=0.1):
        assert s.index[0] == cross_id or s.index[-1] == cross_id, 'cross_id must be the last or first element of the series'
        if len(s) < 2:
            return cross_id
        if s.index[-1] == cross_id:
            s = s.copy().iloc[::-1]
        i = 1  # start after cross_id
        grace_count = 0
        while grace_count <= grace and i < len(s)-1:
            if s.iloc[i] < dist_threshold:
                grace_count += 1
            else:
                grace_count = 0
            i += 1
        return s.index[i]

    def cluster_trajectories(self, pose: pd.DataFrame, cross_y_val=4, frames_around_cross=200, window_length=31,
                             min_traj_len=3, only_to_screen=False):
        trajs = {}
        dist_df = pose[['time', 'x', 'y', 'prob']].reset_index().copy().rename(columns={'index': 'frame_id'})
        dist_df = dist_df.drop(index=dist_df[(dist_df.prob < 0.5) | (dist_df.x.isna())].index, errors='ignore')
        if len(dist_df) < window_length:
            return trajs
        dist_df['time_diff'] = dist_df.time.diff().dt.total_seconds()
        dists = dist_df.y.diff()
        dist_df['distance'] = savgol_filter(dists.values, window_length=window_length, polyorder=0, mode='interp')
        g = dist_df.groupby(dist_df['time_diff'].ge(1).cumsum())

        # cols = 5
        # rows = int(np.ceil(len(g)/cols))
        # fig, axes = plt.subplots(rows, cols, figsize=(25, 3*rows))
        # axes = axes.flatten()
        for i, (group_id, xf) in enumerate(g):
            xf = xf.query('y>-1')
            df = xf.query(f'{cross_y_val - 1}<=y<={cross_y_val + 1}')
            g2 = df.groupby(df.index.to_series().diff().ge(60).cumsum())
            crosses = []
            # axes[i].plot(xf.y)
            for n, gf in g2:
                cross_id = (gf.y - cross_y_val).abs().idxmin()
                lower_lim = self.get_cross_limit(cross_id, xf.loc[cross_id-frames_around_cross:cross_id, 'distance'])
                upper_lim = self.get_cross_limit(cross_id, xf.loc[cross_id:cross_id+frames_around_cross, 'distance'])
                dy_traj = np.abs(xf.loc[upper_lim, 'y'] - xf.loc[lower_lim, 'y'])
                if dy_traj < min_traj_len or (only_to_screen and xf.loc[upper_lim, 'y'] > 3):
                    continue
                crosses.append(cross_id)
                frame_id = xf.loc[lower_lim, 'frame_id']
                trajs[frame_id] = xf.loc[lower_lim:upper_lim, ['x', 'y']].copy()

        #     axes[i].scatter(x=crosses, y=xf.loc[crosses, 'y'], c='red')
        #     axes[i].set_title(f'#Crosses={len(crosses)}')
        # fig.tight_layout()
        # plt.show()

        # dists = np.sqrt(pose.x.diff() ** 2 + pose.y.diff() ** 2).dropna()
        # indices = dists.index.tolist()
        # vs = savgol_filter(dists.values, window_length=31, polyorder=0, mode='interp')
        # current_group = None
        # nan_counter = 0
        # t = (pose.time - pose.time.iloc[0]).dt.total_seconds()
        # for i, pose_i in enumerate(indices):
        #     pos = (pose.x.loc[pose_i], pose.y.loc[pose_i])
        #     if vs[i] > 4:
        #         if not current_group:
        #             current_group = pose_i
        #         trajs.setdefault(current_group, []).append(pos)
        #         nan_counter = 0
        #     elif np.isnan(vs[i]) and nan_counter <= max_nan_seq:
        #         nan_counter += 1
        #         continue
        #     else:
        #         if current_group:
        #             if self.calc_traj_distance(trajs[current_group]) < 3:
        #                 del trajs[current_group]
        #         current_group = None
        #         nan_counter = 0
        #
        # ax = plt.subplot()
        # ax.plot(t[1:], vs)
        # for start_frame, traj in trajs.items():
        #     traj = np.array(traj)
        #     rect = patches.Rectangle((t[start_frame], 0), t[start_frame+len(traj)] - t[start_frame], 5, linewidth=1, facecolor='g', alpha=0.4)
        #     ax.add_patch(rect)
        # plt.show()

        return trajs

    def find_crosses(self, video_path=None, y_value=10, is_play=True, axes=None, cols=3):
        if video_path is not None:
            pose_dict = {'video_crosses': self._load_pose(video_path)}
        else:
            pose_dict = self.get_pose()

        axes_ = self.get_axes(cols, len(pose_dict), axes)
        x_values = {}
        for i, (group_name, pose_df) in enumerate(pose_dict.items()):
            df_ = pose_df.query(f'{y_value-0.1} <= y <= {y_value+0.1} and 0 <= x <= 40').copy()
            m = df_.index.to_series().diff().ne(1).cumsum()
            idx_ = df_.index.to_series().groupby(m).agg(list)
            idx2drop = flatten([idx_[j][1:] for j in idx_[idx_.map(lambda x: len(x)) > 1].index])
            df_.drop(index=idx2drop, inplace=True)

            if is_play and video_path is not None:
                cap = cv2.VideoCapture(video_path)
                for cross_id in df_.index:
                    self.play_segment(cap, cross_id-60, 120, f'cross index {cross_id}')
                cap.release()

            x_values[group_name] = df_.x.values
            sns.kdeplot(x=df_.x.values, ax=axes_[i])
            # axes_[i].hist(df_.x.values, label=f'mean = {df_.x.values.mean():.2f}')
            axes_[i].set_title(group_name)
            axes_[i].legend()

        if len(x_values) == 2:
            groups = list(x_values.values())
            t_stat, p_value = ttest_ind(groups[0], groups[1], equal_var=False)
            p_text = f'p-value={p_value:.3f}' if p_value >= 0.001 else 'p-value<0.001'
            plt.suptitle(f'T={t_stat:.1f}, {p_text}')

        if axes is None:
            plt.tight_layout()
            plt.show()

    @staticmethod
    def calc_traj_distance(traj):
        try:
            traj = np.array(traj)
            traj_no_nan = traj[~np.isnan(traj).any(axis=2), :]
            return distance.euclidean(traj_no_nan[0, :], traj_no_nan[-1, :])

        except:
            return 0
        # return np.sum(np.sqrt(np.sum(np.diff(traj_no_nan, axis=0) ** 2, axis=1)))

    @staticmethod
    def get_axes(cols, n, axes=None, is_cbar=True):
        cols = min(cols, n)
        rows = int(np.ceil(n / cols))
        if axes is None:
            width_ratios = [15 for _ in range(cols)]
            if is_cbar:
                width_ratios += [1]
            fig, axes = plt.subplots(rows, cols+(1 if is_cbar else 0), figsize=(cols * 4, rows * 3),
                                     gridspec_kw={'width_ratios': width_ratios})

        if n > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        return axes


########################################################################################################################


def get_bug_exit_hole(day) -> list:
    orm = ORM()
    with orm.session() as s:
        q = s.query(Block).filter(cast(Block.start_time, Date) == day)
        res = q.all()
    return list(set([r.exit_hole for r in res if r.exit_hole]))


def get_day_from_path(p):
    return p.stem.split('_')[1].split('T')[0]


def get_screen_coords(name):
    s = yaml.load(Path('/analysis/strikes/screen_coords.yaml').open(), Loader=yaml.FullLoader)
    cnt = s['screens'][name]
    return np.array(cnt)


def load_pose_from_videos(animal_id, cam_name):
    orm = ORM()
    dp = ArenaPose(cam_name, 'deeplabcut', is_use_db=True)
    with orm.session() as s:
        for exp in s.query(Experiment).filter_by(animal_id=animal_id).all():
            for blk in exp.blocks:
                for vid in blk.videos:
                    if vid.cam_name != cam_name:
                        continue
                    try:
                        dp.predict_pred_cache(video_path=vid.path)
                    except Exception as exc:
                        print(f'ERROR; {vid.path}; {exc}')


def compare_sides(animal_id='PV80'):
    with ORM().session() as s:
        exps = s.query(Experiment).filter_by(animal_id=animal_id).all()
        days = list(set([e.start_time.strftime('%Y-%m-%d') for e in exps]))

    res = {}
    for day in days.copy():
        exit_holes = [x for x in get_bug_exit_hole(day) if x]
        if len(exit_holes) == 1:
            exit_hole = exit_holes[0]
        else:
            days.remove(day)
            continue
        ps = SpatialAnalyzer(animal_id, day=day).get_pose()
        res.setdefault(exit_hole, []).append(ps)

    res = {k: pd.concat(v).query('y<20') for k, v in res.items()}
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    sa = SpatialAnalyzer(animal_id)
    for i, (k, v) in enumerate(res.items()):
        sa.plot_spatial(v, axes[i])
        axes[i].set_title(k)
    fig.tight_layout()
    plt.show()


def get_videos_to_predict(animal_id=None, experiments_dir=None, model_path=None, is_override=False):
    experiments_dir = experiments_dir or config.EXPERIMENTS_DIR
    p = Path(experiments_dir)
    if animal_id:
        p = p / animal_id
    all_videos = list(p.rglob('*front*.mp4'))
    videos = []
    ap = DLCArenaPose('front', is_use_db=False, model_path=model_path)
    for vid_path in all_videos:
        pred_path = ap.get_predicted_cache_path(vid_path)
        if (not is_override and (pred_path.exists() or pred_path.with_suffix('.txt').exists())) or \
                (len(pred_path.parts) >= 6 and pred_path.parts[-6] == 'test'):
            continue
        videos.append(vid_path)
    videos = sorted(videos, key=lambda x: x.name, reverse=True)
    return videos


def fix_calibrations(animal_id=None, model_path=None):
    videos = get_videos_to_predict(animal_id, model_path=model_path, is_override=True)
    if not videos:
        return
    print(f'found {len(videos)}/{len(videos)} to fix calibration')
    ap = DLCArenaPose('front', is_use_db=False, model_path=model_path)
    is_initialized = False
    for i, video_path in enumerate(videos):
        ap.start_new_session(60)
        if not is_initialized:
            ap.init_from_video(video_path, caliber_only=True)
            is_initialized = True
        try:
            cache_path = ap.get_predicted_cache_path(video_path)
            zf = pd.read_parquet(cache_path)
            for i in tqdm(zf.index, desc=video_path.stem):
                row = zf.loc[i:i].copy()
                new_row = ap.analyze_frame(row['time'].iloc[0], row.copy())
                zf.loc[i] = new_row.iloc[0]
            # zf[['x', 'y']] = zf[['cam_x', 'cam_y']].apply(lambda pos: ap.caliber.get_location(*pos), axis=1).tolist()
            ap.save_predicted_video(zf, video_path)
        except MissingFile as exc:
            print(exc)
        except Exception:
            print(f'\n\n{traceback.format_exc()}\n')


def predict_all_videos(animal_id=None, max_videos=None, experiments_dir=None, model_path=None, errors_cache=None):
    videos = get_videos_to_predict(animal_id, experiments_dir, model_path)
    if not videos:
        return
    print(f'found {len(videos)}/{len(videos)} to predict')
    success_count = 0
    ap = DLCArenaPose('front', is_use_db=False, model_path=model_path)
    for i, video_path in enumerate(videos):
        try:
            if ap.get_predicted_cache_path(video_path).exists():
                continue
            ap.predict_video(video_path=video_path, is_create_example_video=False, prefix=f'({i+1}/{len(videos)}) ')
            success_count += 1
            if max_videos and success_count >= max_videos:
                return
        except MissingFile as exc:
            print_cache(exc, errors_cache)
        except Exception as exc:
            print_cache(exc, errors_cache)


def print_cache(exc, errors_cache):
    if errors_cache is None or str(exc) not in errors_cache:
        print(exc)
    if errors_cache and str(exc) not in errors_cache:
        errors_cache.append(str(exc))


def commit_pose_estimation_to_db(animal_id=None, cam_name='front', min_dist=0.1, min_prob=0.4):
    orm = ORM()
    sa = SpatialAnalyzer(animal_id=animal_id, bodypart='mid_ears', orm=orm, cam_name=cam_name)
    vids = sa.get_videos_paths(is_add_block_video_id=True)
    print(f'Start commit pose of model: {sa.dlc.predictor.model_name}')
    for video_path, block_id, video_id in tqdm(vids['']):
        try:
            pose_df = sa.dlc.load(video_path=video_path, only_load=True).dropna(subset=[('nose', 'x')])
            pose_df = pose_df[(pose_df[[(bp, 'prob') for bp in COMMIT_DB_BODYPARTS]] >= min_prob).any(axis=1)]
            if pose_df.empty:
                continue
            # pose_df['distance'] = np.sqrt((pose_df['nose'][['x', 'y']].diff() ** 2).sum(axis=1))
            # pose_df = pose_df[pose_df[('distance', '')] >= min_dist]
            last_pos = pose_df['nose'][['x', 'y']].iloc[0].values.tolist()
            for i, row in pose_df.iterrows():
                cur_pos = pose_df['nose'][['x', 'y']].loc[i].values.tolist()
                if i != pose_df.index[0] and distance.euclidean(last_pos, cur_pos) < min_dist:
                    continue
                last_pos = cur_pos
                angle = sa.dlc.calc_head_angle(row)
                start_time = datetime.datetime.fromtimestamp(row[('time', '')])
                for bp in COMMIT_DB_BODYPARTS:
                    if np.isnan(row[bp, 'x']):
                        continue
                    orm.commit_pose_estimation(cam_name, start_time, row[(bp, 'x')], row[(bp, 'y')], angle,
                                               None, video_id, sa.dlc.predictor.model_name,
                                               bp, row[(bp, 'prob')], i, animal_id=animal_id, block_id=block_id)
        except Exception as exc:
            print(f'Error: {exc}; {video_path}')


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    # DLCArenaPose('front').test_loaders(1482)
    # print(get_videos_to_predict('PV148'))
    commit_pose_estimation_to_db()
    # predict_all_videos()
    # img = cv2.imread('/data/Pogona_Pursuit/output/calibrations/front/20221205T094015_front.png')
    # plt.imshow(img)
    # plt.show()
    # load_pose_from_videos('PV80', 'front', is_exit_agg=True) #, day='20221211')h
    # SpatialAnalyzer(animal_id='PV91', movement_type='low_horizontal', split_by=['exit_hole'], bodypart='nose').plot_spatial()
    # SpatialAnalyzer(movement_type='low_horizontal', split_by=['exit_hole'], bodypart='nose').find_crosses(y_value=5)
    # SpatialAnalyzer(animal_id='PV91', movement_type='low_horizontal', split_by=['exit_hole'], bodypart='nose').plot_trajectories(only_to_screen=True)
    # SpatialAnalyzer(animal_id='PV91', movement_type='low_horizontal', split_by=['exit_hole'], bodypart='nose').plot_out_of_experiment_pose()
    # fix_calibrations()
    # for vid in sa.get_videos_paths()['exit_hole=bottomLeft']:
    #     sa.play_trajectories(vid, only_to_screen=True)
    # compare_sides(animal_id='PV80')
