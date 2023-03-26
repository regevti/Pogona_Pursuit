import datetime
import math

import yaml
import cv2
import time
import traceback
import importlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from scipy.spatial import distance
from scipy.signal import savgol_filter
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
from analysis.pose_utils import put_text


MIN_DISTANCE = 5  # cm
COMMIT_INTERVAL = 2  # seconds
VELOCITY_SAMPLING_DURATION = 2  # seconds
MODEL_NAME = 'front_head_only_resnet_50'
BAD_VIDEOS = [
    'front_20221130T095549', 'front_20221127T140019', 'front_20221202T163955', 'front_20221122T114015',
    'front_20230314T161544'
]


class MissingFile(Exception):
    """"""


class ArenaPose:
    def __init__(self, cam_name, predictor, is_commit_db=True, orm=None):
        self.cam_name = cam_name
        self.predictor = predictor
        self.is_commit_db = is_commit_db
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

    def load(self, video_path: Path, only_load=False, prefix=''):
        if not self.is_initialized:
            self.initiate_from_video(video_path)
        cache_path = self.get_predicted_cache_path(video_path)
        if cache_path.exists():
            pose_df = pd.read_parquet(cache_path)
        else:
            if not only_load:
                pose_df = self.predict_video(video_path=video_path, prefix=prefix)
            else:
                raise MissingFile(f'Cache file does not exist and only_load=True')
        return pose_df

    def start_new_session(self, fps):
        self.kalman = Kalman(dt=1/fps)
        self.predictions = []

    def load_predictor(self):
        if isinstance(self.predictor, str):
            prd_module, prd_class = config.arena_modules['predictors'][self.predictor]
            prd_module = importlib.import_module(prd_module)
            self.predictor = getattr(prd_module, prd_class)(self.cam_name)

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
        frames_times = self.load_frames_times(db_video_id)

        pose_df = []
        cap = cv2.VideoCapture(video_path)
        n_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(frames_times))
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
        if not db_video_id and video_path:
            db_video_id = self.get_video_db_id(video_path)
        elif not video_path and db_video_id:
            video_path = self.get_video_path(db_video_id)
        video_path = Path(video_path).as_posix()
        return db_video_id, video_path

    def aggregate_to_commit(self, predictions, db_video_id, timestamp):
        predictions = np.array(predictions)
        x, y = [round(z) for z in predictions[:, 1:3].mean(axis=0)]
        if self.is_commit_db and self.is_moved(x, y):
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

    def load_frames_times(self, db_video_id: int) -> pd.DataFrame:
        frames_df = pd.DataFrame()
        with self.orm.session() as s:
            vid = s.query(Video).filter_by(id=db_video_id).first()
            if vid.frames is None:
                return frames_df
            frames_ts = pd.DataFrame(vid.frames.items(), columns=['frame_id', 'time']).set_index('frame_id')
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

    def initiate_from_video(self, video_path: Path):
        """read frame from video and use it to initiate the caliber"""
        cap = cv2.VideoCapture(video_path.as_posix())
        ret, frame = cap.read()
        self.init(frame, caliber_only=True)
        cap.release()

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

    @staticmethod
    def get_predicted_cache_path(video_path) -> Path:
        preds_dir = Path(video_path).parent / 'predictions'
        preds_dir.mkdir(exist_ok=True)
        vid_name = Path(video_path).with_suffix('.parquet').name
        return preds_dir / f'{MODEL_NAME}__{vid_name}'

    def is_moved(self, x, y):
        return not self.last_commit or distance.euclidean(self.last_commit[1:], (x, y)) < MIN_DISTANCE

    def is_ready_to_commit(self, timestamp):
        return self.predictions and (timestamp - self.predictions[0][0]) > COMMIT_INTERVAL


class DLCArenaPose(ArenaPose):
    def __init__(self, cam_name, is_commit_db=True, orm=None, **kwargs):
        super().__init__(cam_name, 'deeplabcut', is_commit_db, orm, **kwargs)
        self.kalman = {}
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
    def __init__(self, animal_id, day=None, cam_name=None):
        self.animal_id = animal_id
        self.day = day
        self.cam_name = cam_name
        self.coords = {
            'arena': np.array([(-3, -2), (55, 78)]),
            'screen': np.array([(-1, -3), (52, -1)])
        }

    def query_pose(self):
        orm = ORM()
        with orm.session() as s:
            q = s.query(PoseEstimation).filter_by(animal_id=self.animal_id)
            if self.cam_name:
                q = q.filter_by(cam_name=self.cam_name)
            q = q.filter(cast(PoseEstimation.start_time, Date) == self.day)
            res = q.all()
        return res

    def get_pose(self):
        res = self.query_pose()
        if not res:
            raise Exception('No pose recordings found')
        pose = pd.DataFrame([(r.x, r.y, r.start_time) for r in res], columns=['x', 'y', 'time'])
        pose = self.drop_out_of_arena_coords(pose)
        return pose

    def drop_out_of_arena_coords(self, df):
        xmin, xmax = self.coords['arena'][:, 0].flatten().tolist()
        idx = df[(df.x < xmin) | (df.x > xmax)].index
        return df.drop(idx)

    def plot_spatial(self, pose=None, ax=None):
        if pose is None:
            pose = self.get_pose()
        if ax is None:
            fig, ax_ = plt.subplots(1, 1, figsize=(10, 20))
        else:
            ax_ = ax
        self.plot_hist2d(pose, ax_)
        self.plot_arena(ax_)
        ax.set_title(f'{self.day} - {",".join(get_bug_exit_hole(self.day))}')
        if ax is None:
            plt.show()

    def plot_arena(self, ax):
        for name, c in self.coords.items():
            rect = patches.Rectangle(c[0, :], *(c[1, :] - c[0, :]).tolist(), linewidth=1, edgecolor='k',
                                     facecolor='k' if name == 'screen' else 'none')
            ax.add_patch(rect)
        ax.invert_xaxis()

    def plot_hist2d(self, df, ax):
        sns.histplot(data=df, x='x', y='y', ax=ax,
                     bins=(30, 30), cmap='Greens', stat='probability', pthresh=.0,
                     cbar=True, cbar_kws=dict(shrink=.75, label='Probability'))

    def plot_trajectories(self, ax=None):
        pose = self.get_pose()
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 20))
        trajs = self.cluster_trajectories(pose)
        for i, traj in trajs.items():
            traj = np.array(traj)
            total_distance = np.sum(np.sqrt(np.sum(np.diff(traj, axis=0) ** 2, axis=1)))
            if traj[0, 1] < 20 or traj[-1, 1] > 20 or total_distance < 5:
                continue

            ax.plot(traj[:, 0], traj[:, 1], label=f'traj{i}')
            ax.scatter(traj[-1, 0], traj[-1, 1], marker='*')
        self.plot_arena(ax)
        plt.legend()
        plt.show()

    def cluster_trajectories(self, pose: pd.DataFrame, dist_thresh=20, time_thresh=5):
        trajs = {}
        last_pos, last_ts, current_group = None, None, None
        v = np.sqrt(pose.x.diff() ** 2 + pose.y.diff() ** 2) / pose.time.diff().dt.total_seconds()
        v = v.values
        vs = savgol_filter(v, window_length=5, polyorder=1, mode='nearest')
        current_group = None
        for i, row in pose.iterrows():
            pos = (row.x, row.y)
            if vs[i] > 100:
                if not current_group:
                    current_group = i
                trajs.setdefault(current_group, []).append(pos)
            else:
                current_group = None
        return trajs

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
    s = yaml.load(Path('/data/Pogona_Pursuit/Arena/analysis/screen_coords.yaml').open(), Loader=yaml.FullLoader)
    cnt = s['screens'][name]
    return np.array(cnt)


def load_pose_from_videos(animal_id, cam_name):
    orm = ORM()
    dp = ArenaPose(cam_name, 'deeplabcut', is_commit_db=True)
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


def foo():
    video_path = Path(f'{config.OUTPUT_DIR}/experiments/PV80/20221211/block1/videos/front_20221211T113717.mp4')
    ap = DLCArenaPose('front', is_commit_db=False)
    pose_df = ap.predict_video(video_path=video_path, is_create_example_video=True)
    return


def convert_all_videos(animal_id=None, max_videos=None):
    p = Path(config.EXPERIMENTS_DIR)
    if animal_id:
        p = p / animal_id
    all_videos = list(p.rglob('*front*.mp4'))
    ap = DLCArenaPose('front', is_commit_db=False)
    ap.initiate_from_video(all_videos[0])
    videos = [v for v in all_videos if not ap.get_predicted_cache_path(v).exists() and v.stem not in BAD_VIDEOS]
    if not videos:
        return
    print(f'found {len(videos)}/{len(all_videos)} to predict')
    success_count = 0
    for i, video_path in enumerate(videos):
        try:
            ap = DLCArenaPose('front', is_commit_db=False)
            if ap.get_predicted_cache_path(video_path).exists():
                continue
            ap.predict_video(video_path=video_path, is_create_example_video=False, prefix=f'({i+1}/{len(videos)}) ')
            success_count += 1
            if max_videos and success_count >= max_videos:
                return
        except MissingFile as exc:
            print(exc)
        except Exception:
            print(f'\n\n{traceback.format_exc()}\n')


if __name__ == '__main__':
    # foo()
    convert_all_videos()
    # img = cv2.imread('/data/Pogona_Pursuit/output/calibrations/front/20221205T094015_front.png')
    # plt.imshow(img)
    # plt.show()
    # load_pose_from_videos('PV80', 'front', is_exit_agg=True) #, day='20221211')
    # ps = SpatialAnalyzer('PV80', day='2022-12-15').get_pose()
    # compare_sides(animal_id='PV80')

