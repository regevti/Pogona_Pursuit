if __name__ == '__main__':
    import os
    os.chdir('..')
from pathlib import Path
import numpy as np
from datetime import datetime
from functools import cached_property, lru_cache
import pandas as pd
import pickle
import cv2
from tqdm.auto import tqdm
from scipy.signal import find_peaks, savgol_filter
import matplotlib
import matplotlib.pyplot as plt
from analysis.pose_utils import closest_index, pixels2cm, distance, remove_outliers
from analysis.pose import DLCPose, PredPlotter
from utils import KalmanFilter
from db_models import ORM, Experiment, Block, Video, VideoPrediction, Strike, Trial
from calibration import PoseEstimator
from analysis.tongue_out import TongueDetector, TONGUE_CLASS, TONGUE_PREDICTED_DIR

NUM_FRAMES_TO_PLOT = 5
NUM_POSE_FRAMES_PER_STRIKE = 30
PREDICTOR_NAME = 'pogona_head_local'
TONGUE_COL = ('tongue', '')


class StrikeException(Exception):
    """"""


class MissingStrikeData(Exception):
    """Could not find timestamps for video frames"""


class Loader:
    def __init__(self, strike_db_id, cam_name, is_load_pose=True):
        self.strike_db_id = strike_db_id
        self.cam_name = cam_name
        self.is_load_pose = is_load_pose
        self.orm = ORM()
        self.bug_traj_strike_id = None
        self.strike_frame_id = None
        self.video_path = None
        self.frames_df: pd.DataFrame = pd.DataFrame()
        self.traj_df: pd.DataFrame = pd.DataFrame(columns=['time', 'x', 'y'])
        self.info = {}
        self.load(is_load_pose)

    def __str__(self):
        return f'Strike-Loader:{self.strike_db_id}'

    def load(self, is_load_pose):
        with self.orm.session() as s:
            strk = s.query(Strike).filter_by(id=self.strike_db_id).first()
            if strk is None:
                raise StrikeException(f'could not find strike id: {self.strike_db_id}')

            self.info = {k: v for k, v in strk.__dict__.items() if not k.startswith('_')}
            trial = s.query(Trial).filter_by(id=strk.trial_id).first()
            if trial is None:
                raise StrikeException('No trial found in DB')

            self.traj_df = pd.DataFrame(trial.bug_trajectory)
            self.traj_df['time'] = pd.to_datetime(self.traj_df.time).dt.tz_localize(None)
            block = s.query(Block).filter_by(id=trial.block_id).first()
            for vid in block.videos:
                if vid.cam_name != self.cam_name:
                    continue
                video_path = Path(vid.path).resolve()
                assert video_path.exists(), f'Video {video_path} does not exist'
                self.video_path = video_path
                self.frames_df = self.load_frames_times(vid)
                if is_load_pose and not self.frames_df.empty and \
                        (self.frames_df.iloc[0].time <= strk.time <= self.frames_df.iloc[-1].time):
                    self.load_pose_df(video_path)
                    break
            if self.frames_df.empty:
                raise MissingStrikeData()

            self.strike_frame_id = (strk.time - self.frames_df.time).dt.total_seconds().abs().idxmin()
            self.bug_traj_strike_id = (strk.time - self.traj_df.time).dt.total_seconds().abs().idxmin()
            self.load_tongues_out()

    def load_frames_times(self, vid) -> pd.DataFrame:
        if vid.frames is None:
            print(f'{str(self)} - frame times does not exist')
            return self.frames_df
        frames_ts = pd.DataFrame(vid.frames.items(), columns=['frame_id', 'time']).set_index('frame_id')
        frames_ts['time'] = pd.to_datetime(frames_ts.time, unit='s', utc=True).dt.tz_convert('Asia/Jerusalem').dt.tz_localize(None)
        frames_ts.index = frames_ts.index.astype(int)
        return frames_ts

    def load_pose_df(self, video_path: Path):
        pred_path = DLCPose.get_cache_path(video_path)
        if not pred_path.exists():
            raise MissingStrikeData(f'could not find pose predictions file in {pred_path}')
        pdf = pd.read_parquet(pred_path)
        frames_df_ = self.frames_df.copy()
        if len(frames_df_) > len(pdf):
            frames_df_ = frames_df_.iloc[:len(pdf)]
        elif 1 <= (len(pdf) - len(frames_df_)) <= 10:
            pdf = pdf.iloc[:len(frames_df_)]
        if len(frames_df_) != len(pdf):
            raise MissingStrikeData(f'different length to frames_df ({len(frames_df_)}) and pose_df ({len(pdf)})')
        frames_df_.columns = pd.MultiIndex.from_tuples([('time', '')])
        self.frames_df = pd.concat([frames_df_, pdf], axis=1)

    def get_strike_frame(self) -> np.ndarray:
        for _, frame in self.gen_frames_around_strike(0, 1):
            return frame

    def gen_frames_around_strike(self, n_frames_back=100, n_frames_forward=20, center_frame=None):
        center_frame = center_frame or self.strike_frame_id
        start_frame = center_frame - n_frames_back
        cap = cv2.VideoCapture(self.video_path.as_posix())
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for i in range(start_frame, start_frame + n_frames_back + n_frames_forward):
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yield i, frame
        cap.release()

    def load_tongues_out(self):
        td = TongueDetector(use_cv2_resize=True, threshold=0.9).inference_init()
        self.frames_df[TONGUE_COL] = [0] * len(self.frames_df)

        for i, frame in self.gen_frames_around_strike():
            label, _ = td.predict_image(frame)
            if label == TONGUE_CLASS:
                self.frames_df.loc[i, [TONGUE_COL]] = 1
                # cv2.imwrite(f'{TONGUE_PREDICTED_DIR}/{self.video_path.stem}_{i}.jpg', frame)

    def plot_strike_events(self, n_frames_back=100, n_frames_forward=20):
        plt.figure()
        plt.axvline(self.strike_frame_id, color='r')
        start_frame = self.strike_frame_id - n_frames_back
        end_frame = start_frame + n_frames_back + n_frames_forward
        plt.xlim([start_frame, end_frame])
        for i in range(start_frame, end_frame):
            tongue_val = self.frames_df[TONGUE_COL][i]
            if tongue_val == 1:
                plt.axvline(i, linestyle='--', color='b')
        plt.title(str(self))
        plt.show()

    def play_strike(self, n_frames_back=100, n_frames_forward=20, annotations=None):
        for i, frame in self.gen_frames_around_strike(n_frames_back, n_frames_forward):
            if i == self.strike_frame_id:
                PredPlotter.put_text('Strike Frame', frame, 30, 20)
            if annotations and i in annotations:
                PredPlotter.put_text(annotations[i], frame, 30, frame.shape[0]-30)
            if self.is_load_pose:
                PredPlotter.plot_predictions(frame, i, self.frames_df)
            frame = cv2.resize(frame, None, None, fx=0.5, fy=0.5)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow(str(self), frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


class StrikeAnalyzer:
    def __init__(self, loader: Loader = None, payload: dict = None, pose_df: pd.DataFrame = None,
                 bug_traj: pd.DataFrame = None):
        """
        Payload fields:
        - time: datetime
        - x,y : strike position
        - bug_x,bug_y : bug position at strike time
        - is_hit: boolean
        - is_reward_any_touch: boolean. Reward should be given due to any-touch probability
        - is_reward_bug: boolean
        - is_climbing: boolean
        - bug_type: string
        - bug_size: integer
        - in_block_trial_id: integer. The trial number inside its block
        - trial_id: trial DB index

        Pose-DF columns:
        - frame_id
        - time
        - <key-point> / x, y, cam_x, cam_y, prob : float

        Bug Trajectory columns:
        - time
        - x
        - y
        """
        self.loader = loader
        self.payload = payload
        self.pose_df = pose_df
        self.bug_traj = bug_traj
        img = self.loader.get_strike_frame()
        self.caliber = PoseEstimator(self.loader.cam_name, is_debug=False).init(img)
        self.check_arguments()
        self.strike_position = (self.payload.get('x'), self.payload.get('y'))

    def check_arguments(self):
        if self.loader is not None:
            if self.payload is None:
                self.payload = self.loader.info
            if self.pose_df is None:
                fdf = self.loader.frames_df.copy()
                self.pose_df = pd.concat([fdf['time'], fdf['nose']], axis=1)
                self.pose_df[['x', 'y']] = self.pose_df[['cam_x', 'cam_y']].apply(
                    lambda pos: self.caliber.get_location(*pos), axis=1).tolist()
                # smoothing of x and y
                for c in ['x', 'y']:
                    self.pose_df[c] = savgol_filter(self.pose_df[c], window_length=37, polyorder=0)
                    # self.pose_df[c] = remove_outliers(self.pose_df.set_index('time')[c].rolling('100ms').mean(std=5).values, is_replace_nan=True)

                # ax = plt.subplot()
                # y = self.pose_df.set_index('time')['y']
                # ax.plot(y, label=f'orig ({len(y)})')
                # y2 = y.rolling('100ms').mean(std=5)
                # y2 = remove_outliers(y2)
                # ax.plot(y2, label=f'new ({len(y2)})')
                # ax.legend()
                # plt.show()

            if self.bug_traj is None:
                self.bug_traj = self.loader.traj_df

        assert self.pose_df is not None and len(self.pose_df) > 1
        mandatory_fields = ['time', 'x', 'y', 'bug_x', 'bug_y', 'is_hit']
        for fld in mandatory_fields:
            assert fld in self.payload, f'Field {fld} was not found in strike payload'
        # check frame is in pose_df and bug_traj
        # assert self.pose_df.time

    def plot_strike_analysis(self, n=6, only_save_to=None, only_return=False):
        # get tongue frame IDs
        fig = plt.figure(figsize=(20, 15))
        grid = fig.add_gridspec(ncols=n, nrows=3, width_ratios=[1]*n, height_ratios=[3, 3, 5], hspace=0.2)
        self.plot_frames_sequence(grid[:2, :], n)
        self.plot_nose_pose(fig.add_subplot(grid[2, :2]))
        self.plot_nose_velocity(fig.add_subplot(grid[2, 2:4]))
        self.plot_projected_strike(fig.add_subplot(grid[2, 4:]))
        if only_save_to:
            path = Path(only_save_to)
            assert path.is_dir()
            fig.savefig(path / f'{self.loader.strike_db_id}.jpg')
            plt.close(fig)
        elif only_return:
            fig.canvas.draw()
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return image_from_plot
        else:
            plt.show()

    def plot_frames_sequence(self, grid, n):
        tng = self.loader.frames_df['tongue']
        tongue_frame_ids = tng[tng == 1].index.tolist()
        inner_grid = grid.subgridspec(2, n, wspace=0.1, hspace=0.2)
        axes = inner_grid.subplots()  # Create all subplots for the inner grid.
        axes = axes.flatten()
        for i, (frame_id, frame) in enumerate(self.loader.gen_frames_around_strike(n, n, center_frame=self.calc_strike_frame)):
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            frame = PredPlotter.plot_predictions(frame, frame_id, self.loader.frames_df, parts2plot=['nose'])
            h, w = frame.shape[:2]
            frame = frame[h//2:, 350:w-350]
            labels = []
            if frame_id == self.strike_frame_id:
                labels.append('strike frame')
            if frame_id == self.calc_strike_frame:
                labels.append('pose-calculated strike')
            if frame_id in tongue_frame_ids:
                labels.append('tongue')
            if frame_id == self.leap_frame:
                labels.append('leap frame')

            h_text, text_colors = 30, {'strike frame': (255, 0, 0), 'pose-calculated strike': (255, 255*0.647, 0),
                                       'tongue': (0, 255, 0), 'leap frame': (0,0,255)}
            for lbl in labels:
                frame = cv2.putText(frame, lbl, (20, h_text), cv2.FONT_HERSHEY_PLAIN, 3,
                                    text_colors[lbl], 2, cv2.LINE_AA)
                h_text += 30
            axes[i].imshow(frame)
            axes[i].set_xticks([]); axes[i].set_yticks([])
            axes[i].set_title(frame_id)
            # axes[i].set_title('\n'.join(labels))

    def plot_nose_pose(self, ax, n=100):
        y, t, frames_ids = self._get_pose_y_and_t(n)
        ax.plot(t, y, color='k')
        self._plot_strikes_lines(frames_ids, ax, np.mean(np.diff(t)))
        ax.set_title('Nose position vs. frame_ids')
        ax.set_xlabel('Time around strike [sec]')
        ax.set_ylabel('Nose Y-value [cm]')
        ax.legend()

    def plot_nose_velocity(self, ax, n=100):
        y, t, frames_ids = self._get_pose_y_and_t(n)
        dy, dt = np.diff(y), np.diff(t)
        ax.plot(t[1:], dy/dt, color='k')
        self._plot_strikes_lines(frames_ids, ax, np.mean(dt))
        ax.set_title('Nose Velocity vs. frame_ids')
        ax.set_xlabel('Time around strike [sec]')
        ax.set_ylabel('Nose Velocity [cm/sec]')
        ax.legend()

    def _get_pose_y_and_t(self, n):
        frames_ids = np.arange(self.calc_strike_frame - n, self.calc_strike_frame + n + 1)
        df = self.pose_df.loc[frames_ids.tolist()]
        # df = df.set_index('time')['nose']['y'].rolling('100ms').sum()
        t = df.time.view('int64').values / 1e9
        t = t - t[self.strike_frame_id - frames_ids[0]]
        # kalman = KalmanFilter()
        # y = np.array([kalman.get_filtered((row.x, row.y)) for i, row in df['nose'][['x', 'y']].iterrows()])[:, 1]
        y = df['y'].values
        return y, t, frames_ids

    def _plot_strikes_lines(self, frames, ax, dt):
        if self.leap_frame in frames:
            ax.axvline(dt * (self.leap_frame - self.calc_strike_frame), linestyle='--', color='b', label='leap_frame')
        if self.strike_frame_id in frames:
            ax.axvline(dt * (self.strike_frame_id - self.calc_strike_frame), linestyle='--', color='r', label='strike_frame')
        if self.calc_strike_frame in frames:
            ax.axvline(0, linestyle='--', color='orange', label='calc strike frame')
        ax.axhline(0)

    def plot_projected_strike(self, ax, is_plot_strike_only=False):
        try:
            if self.proj_strike_pos is not None:
                ax.scatter(*self.proj_strike_pos, label='strike', color='r', marker='X')
            if is_plot_strike_only:
                return
            ax.scatter(0, 0, label='bug position at strike', color='m', marker='D')
            bug_radius = self.payload.get('bug_size')
            if self.leap_frame and self.proj_leap_pos is not None:
                ax.scatter(*self.proj_leap_pos, label='bug position at leap', color='g', marker='D')
            ax.add_patch(plt.Circle((0, 0), bug_radius, color='lemonchiffon', alpha=0.4))
            ax.legend()
            if self.proj_leap_pos is not None and self.proj_strike_pos is not None:
                lim = max([*np.abs(self.proj_strike_pos), *np.abs(self.proj_leap_pos), bug_radius])
                ax.set_xlim([-lim, lim])
                ax.set_ylim([-lim, lim])
                ax.plot([0, 0], [-lim, lim], 'k')
                ax.plot([-lim, lim], [0, 0], 'k')
            title = 'Projected Strike Coordinates'
            if self.prediction_distance:
                title += f'\nPD: {self.prediction_distance:.2f} cm'
            if self.bug_speed:
                title += f', BugSpeed: {self.bug_speed:.1f} cm/sec'
            ax.set_title(title)
        except ImportError as exc:
            print(f'Error plotting projected strike; {exc}')

    @cached_property
    def proj_strike_pos(self):
        proj_strike, _ = self.get_projected_coords()
        return proj_strike

    @cached_property
    def proj_leap_pos(self):
        _, proj_leap = self.get_projected_coords()
        return proj_leap

    def get_projected_coords(self, default_leap=20):
        leap_frame = self.leap_frame or self.calc_strike_frame - default_leap
        leap_pos = self.get_bug_pos_at_frame(leap_frame)
        if leap_pos is None:
            return None, None
        else:
            return self.project_strike_coordinates(leap_pos.flatten())

    def project_strike_coordinates(self, leap_pos):
        """Project bug trajectory space to the vector spanned from bug position at leap to bug position at strike"""
        bug_pos = self.get_bug_pos_at_frame(self.calc_strike_frame).flatten()
        strike_pos = np.array([self.payload['x'], self.payload['y']])
        u = bug_pos - leap_pos
        xr = np.array([1, 0])
        th = np.arctan2(np.linalg.det([u, xr]), np.dot(u, xr))
        r = np.array(((np.cos(th), -np.sin(th)),
                      (np.sin(th), np.cos(th))))
        r0 = r.dot(bug_pos)
        return r.dot(strike_pos) - r0, r.dot(leap_pos) - r0

    def get_bug_pos_at_frame(self, frame_id, n_back=0):
        i = closest_index(self.bug_traj.time, self.pose_df.loc[frame_id, 'time'])
        if not i:
            return
        return self.bug_traj.loc[i-n_back:i, ['x', 'y']].values.astype('float')

    @cached_property
    def bug_traj_last_frame(self):
        return closest_index(self.pose_df.time, self.bug_traj.time.iloc[-1])

    @cached_property
    def strike_frame_id(self):
        return closest_index(self.pose_df.time, self.payload.get('time'))

    @cached_property
    def leap_frame(self):
        y = self.pose_df.y
        dy = y.diff()
        leap_frame_idx = y.index[0]
        grace_count = 0
        for r in np.arange(self.calc_strike_frame, dy.index[0], -1):
            if dy[r] > -0.03:
                if grace_count == 0:
                    leap_frame_idx = r
                grace_count += 1
                if grace_count < 5:
                    continue
                break
            else:
                grace_count = 0
        return leap_frame_idx

    @cached_property
    def calc_strike_frame(self):
        max_diff_strike_frame = 100
        y = self.pose_df.y.values
        peaks_idx, _ = find_peaks(-y, height=0.5, distance=10)
        peaks_idx = peaks_idx[(np.abs(peaks_idx - self.strike_frame_id) < max_diff_strike_frame) &
                              (peaks_idx < self.bug_traj_last_frame)]

        strike_frame_idx = None
        try:
            if len(peaks_idx) > 0:
                strike_frame_idx = peaks_idx[np.argmin(y[peaks_idx])]
            else:
                yrange = np.arange(self.strike_frame_id - max_diff_strike_frame,
                                   self.strike_frame_id + max_diff_strike_frame)
                around_strike_y_values = [y[idx] for idx in yrange if idx in np.arange(len(y)).astype(int)]
                if len(around_strike_y_values) > 0:
                    strike_frame_idx = yrange[np.argmin(y[yrange])]
        except Exception as exc:
            print(f'Error in calc_strike_frame: {exc}')
        return strike_frame_idx

    @cached_property
    def prediction_distance(self):
        if self.leap_frame is None:
            return
        if self.bug_traj is None:
            return
        traj_id = closest_index(self.bug_traj.time, self.pose_df.loc[self.leap_frame, 'time'])
        bug_pos = self.bug_traj.loc[traj_id, ['x', 'y']].values.tolist()
        d = distance(*bug_pos, *self.strike_position)
        return pixels2cm(d)

    @cached_property
    def bug_speed(self):
        d = self.bug_traj.diff().iloc[1:, :]
        v = remove_outliers(np.sqrt(d.x ** 2 + d.y ** 2) / d.time.dt.total_seconds())
        return pixels2cm(v.mean())  # speed in cm/sec

    @cached_property
    def speed_group(self):
        if 2 <= self.bug_speed < 6:
            return 4
        elif 6 <= self.bug_speed < 10:
            return 8
        elif 10 <= self.bug_speed < 14:
            return 12

    @property
    def is_hit(self):
        return self.payload.get('is_hit', False)


class MultiTrialAnalysis:
    def __init__(self, **filters):
        self.filters = filters
        self.orm = ORM()

    def load_strikes(self):
        strikes_ids = []
        with self.orm.session() as s:
            blocks = s.query(Block).filter_by(**self.filters).all()
            for blk in blocks:
                strikes_ids.extend([strk.id for strk in blk.strikes if not strk.is_climbing])
        return strikes_ids


class CircleMultiTrialAnalysis(MultiTrialAnalysis):
    def __init__(self, is_use_cache=True, **filters):
        super().__init__(**filters)
        self.is_use_cache = is_use_cache
        self.filters['movement_type'] = 'circle'
        self.df = []
        self.load_data()

    def load_data(self):
        sids = self.load_strikes()
        cache_df = pd.read_parquet(self.cache_path)
        for sid in tqdm(sids, desc='loading data'):
            if self.is_use_cache and sid in cache_df.strike_id.values:
                self.df.append(cache_df.query(f'strike_id=={sid}').iloc[0].to_dict())
            else:
                try:
                    ld = Loader(sid, 'front')
                    sa = StrikeAnalyzer(ld)
                    bug_pos = sa.get_bug_pos_at_frame(sa.calc_strike_frame).flatten()
                    strike_pos = sa.payload['x'], sa.payload['y']
                    self.df.append({'strike_id': sid, 'bug_pos_x': bug_pos[0], 'bug_pos_y': bug_pos[1],
                                    'strike_pos_x': strike_pos[0], 'strike_pos_y': strike_pos[1]})
                except MissingStrikeData:
                    continue
                except Exception as exc:
                    print(f'Error strike-{sid}: {exc}')
        self.df = pd.DataFrame(self.df)
        self.df.to_parquet(self.cache_path)

    @property
    def cache_path(self):
        return Path('/data/Pogona_Pursuit/output/strike_analysis/multi_trial') / 'circle.parquet'

    def plot_circle_strikes(self):
        plt.scatter(self.df.bug_pos_x, self.df.bug_pos_y, c='g')
        plt.scatter(self.df.strike_pos_x, self.df.strike_pos_y, c='r')
        plt.gca().invert_yaxis()
        plt.show()


def delete_duplicate_strikes(animal_id):
    orm = ORM()
    with orm.session() as s:
        for exp in s.query(Experiment).filter_by(animal_id=animal_id).all():
            for blk in exp.blocks:
                if not blk.strikes:
                    continue
                strikes = sorted(blk.strikes, key=lambda x: x.time)
                diffs = np.diff([st.time for st in strikes])
                for i, strk in enumerate(strikes):
                    if i > 0 and diffs[i-1].total_seconds() == 0:
                        s.delete(strk)
                        print(f'Deleting strike {strk.time} from {exp.name}, block{blk.block_id}')
                        continue
        s.commit()


def load_strikes(animal_id, start_time=None):
    orm = ORM()
    strikes_ids = []
    with orm.session() as s:
        exps = s.query(Experiment).filter_by(animal_id=animal_id)
        if start_time:
            exps = exps.filter(Experiment.start_time >= start_time)
        for exp in exps.all():
            for blk in exp.blocks:
                for strk in blk.strikes:
                    if strk.is_climbing:
                        continue
                    strikes_ids.append(strk.id)
    return strikes_ids


def play_strikes(animal_id, start_time=None, cam_name='front', is_load_pose=True, strikes_ids=None):
    if not strikes_ids:
        strikes_ids = load_strikes(animal_id, start_time)
    for sid in strikes_ids:
        try:
            ld = Loader(sid, cam_name, is_load_pose=is_load_pose)
            ld.play_strike()
        except Exception as exc:
            print(f'ERROR strike_id={sid}: {exc}')


def analyze_strikes(animal_id, cam_name='front', n_frames_back=50, start_time=None, strike_id=None):
    if strike_id:
        strikes_ids = [strike_id]
    else:
        strikes_ids = load_strikes(animal_id, start_time=start_time)
        strikes_ids = sorted(strikes_ids)
    orm = ORM()
    # strikes_ids = [6212, 5982, 6092, 6250, 6561]
    with tqdm(strikes_ids) as pbar:
        for sid in pbar:
            pbar.desc = f'Strike-{sid}'
            try:
                ld = Loader(sid, cam_name)
                # ld.play_strike()
                # ld.plot_strike_events()
                sa = StrikeAnalyzer(ld)
                # sa.plot_strike_analysis(only_save_to='/data/Pogona_Pursuit/output/strike_analysis/PV80')
                with orm.session() as s:
                    strk = s.query(Strike).filter_by(id=sid).first()
                    strk.prediction_distance = sa.prediction_distance
                    strk.calc_speed = sa.bug_speed
                    strk.projected_strike_coords = sa.proj_strike_pos.tolist()
                    strk.projected_leap_coords = sa.proj_leap_pos.tolist()
                    s.commit()
            except MissingStrikeData:
                print(f'Strike-{sid}: No timestamps for frames')
            except Exception as exc:
                print(f'Strike-{sid}: {exc}')


if __name__ == '__main__':
    CircleMultiTrialAnalysis().plot_circle_strikes()
    # ld = Loader(5968, 'front')
    # sa = StrikeAnalyzer(ld)
    # sa.plot_strike_analysis()
    # delete_duplicate_strikes('PV80')
    # play_strikes('PV80', start_time='2022-12-01', cam_name='front', is_load_pose=False, strikes_ids=[6365])
    # analyze_strikes('PV80')
    # foo()
    # calibrate()
    # save_strikes_dataset('/data/Pogona_Pursuit/output/datasets/pogona_tongue/', 'PV80')
