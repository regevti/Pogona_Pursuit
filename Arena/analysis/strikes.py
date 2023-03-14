import time
from pathlib import Path
import numpy as np
from datetime import datetime
from functools import cached_property, lru_cache
import pandas as pd
import pickle
import cv2
import yaml
from tqdm.auto import tqdm
from scipy.signal import find_peaks, savgol_filter
import matplotlib
if __name__ == '__main__':
    import os
    os.chdir('..')

import matplotlib.pyplot as plt
import seaborn as sns
from analysis.pose_utils import closest_index, pixels2cm, distance, remove_outliers, fit_circle
from analysis.pose import ArenaPose
# from analysis.predictors.deeplabcut import PredPlotter, DLCPose, MODEL_NAME as POSE_MODEL_NAME
from db_models import ORM, Experiment, Block, Video, VideoPrediction, Strike, Trial
from calibration import CharucoEstimator
from analysis.predictors.tongue_out import TONGUE_CLASS, TONGUE_PREDICTED_DIR, TongueOutAnalyzer

NUM_FRAMES_TO_PLOT = 5
NUM_POSE_FRAMES_PER_STRIKE = 30
PREDICTOR_NAME = 'pogona_head_local'
TONGUE_COL = ('tongue', '')


class StrikeException(Exception):
    """"""


class MissingStrikeData(Exception):
    """Could not find timestamps for video frames"""


class Loader:
    def __init__(self, strike_db_id, cam_name, is_load_pose=True, is_debug=True, is_load_tongue=True, orm=None):
        self.strike_db_id = strike_db_id
        self.cam_name = cam_name
        self.is_load_pose = is_load_pose
        self.is_load_tongue = is_load_tongue
        self.is_debug = is_debug
        self.orm = orm if orm is not None else ORM()
        self.sec_back = 3
        self.sec_after = 2
        self.frames_delta = None
        self.n_frames_back = None
        self.n_frames_forward = None
        self.dlc_pose = ArenaPose(cam_name, 'deeplabcut', is_commit_db=False, orm=orm)
        self.bug_traj_strike_id = None
        self.bug_traj_before_strike = None
        self.strike_frame_id = None
        self.video_path = None
        self.frames_df: pd.DataFrame = pd.DataFrame()
        self.traj_df: pd.DataFrame = pd.DataFrame(columns=['time', 'x', 'y'])
        self.info = {}
        self.load()

    def __str__(self):
        return f'Strike-Loader:{self.strike_db_id}'

    def load(self):
        with self.orm.session() as s:
            n_tries = 3
            for i in range(n_tries):
                try:
                    strk = s.query(Strike).filter_by(id=self.strike_db_id).first()
                    break
                except Exception as exc:
                    time.sleep(0.2)
                    if i >= n_tries - 1:
                        raise exc
            if strk is None:
                raise StrikeException(f'could not find strike id: {self.strike_db_id}')

            self.info = {k: v for k, v in strk.__dict__.items() if not k.startswith('_')}
            trial = s.query(Trial).filter_by(id=strk.trial_id).first()
            if trial is None:
                raise StrikeException('No trial found in DB')

            self.load_bug_trajectory_data(trial, strk)
            self.load_frames_data(s, trial, strk)
            self.load_tongues_out()

    def load_bug_trajectory_data(self, trial, strk):
        self.traj_df = pd.DataFrame(trial.bug_trajectory)
        self.traj_df['time'] = pd.to_datetime(self.traj_df.time).dt.tz_localize(None)
        self.bug_traj_strike_id = (strk.time - self.traj_df.time).dt.total_seconds().abs().idxmin()

        n = self.sec_back / self.traj_df['time'].diff().dt.total_seconds().mean()
        self.bug_traj_before_strike = self.traj_df.loc[self.bug_traj_strike_id-n:self.bug_traj_strike_id].copy()

    def load_frames_data(self, s, trial, strk):
        block = s.query(Block).filter_by(id=trial.block_id).first()
        for vid in block.videos:
            if vid.cam_name != self.cam_name:
                continue
            video_path = Path(vid.path).resolve()
            if not video_path.exists():
                print(f'Video path does not exist: {video_path}')
                continue
            frames_times = self.load_frames_times(vid)
            # check whether strike's time is in the loaded frames_times
            if not frames_times.empty and \
                    (frames_times.iloc[0].time <= strk.time <= frames_times.iloc[-1].time):
                # if load pose isn't needed finish here
                self.strike_frame_id = (strk.time - frames_times.time).dt.total_seconds().abs().idxmin()
                if not self.is_load_pose:
                    self.frames_df = frames_times
                # otherwise, load all pose data around strike frame
                else:
                    try:
                        self.load_pose(video_path, frames_times)
                    except Exception as exc:
                        raise MissingStrikeData(str(exc))
                # break since the relevant video was found
                self.video_path = video_path
                break
            # if strike's time not in frames_times continue to the next video
            else:
                continue

        if self.frames_df.empty:
            raise MissingStrikeData()

    def load_frames_times(self, vid):
        frames_times = self.dlc_pose.load_frames_times(vid.id)
        if not frames_times.empty:
            self.frames_delta = np.mean(frames_times.time.diff().dt.total_seconds())
            self.n_frames_back = round(self.sec_back / self.frames_delta)
            self.n_frames_forward = round(self.sec_after / self.frames_delta)
        return frames_times

    def get_strike_frame(self) -> np.ndarray:
        for _, frame in self.gen_frames_around_strike(0, 1):
            return frame

    def gen_frames(self, frame_ids):
        cap = cv2.VideoCapture(self.video_path.as_posix())
        start_frame, end_frame = frame_ids[0], frame_ids[-1]
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for i in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if i not in frame_ids:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yield i, frame
        cap.release()

    def gen_frames_around_strike(self, n_frames_back=None, n_frames_forward=None, center_frame=None, step=1):
        n_frames_back, n_frames_forward = n_frames_back or self.n_frames_back, n_frames_forward or self.n_frames_forward
        center_frame = center_frame or self.strike_frame_id
        start_frame = center_frame - (n_frames_back * step)
        frame_ids = [i for i in range(start_frame, start_frame + step * (n_frames_back + n_frames_forward), step)]
        return self.gen_frames(frame_ids)

    def load_pose(self, video_path, frames_times):
        frames_range = (self.strike_frame_id - self.n_frames_back, self.strike_frame_id + self.n_frames_forward)
        pose_df = self.dlc_pose.load(video_path.as_posix(), frames_range=frames_range, is_debug=False)
        self.frames_df = pd.merge(frames_times, pose_df, how='left')

    def load_tongues_out(self):
        if not self.is_load_tongue:
            return
        toa = TongueOutAnalyzer(is_debug=self.is_debug)
        self.frames_df[TONGUE_COL] = [0] * len(self.frames_df)

        for i, frame in self.gen_frames_around_strike():
            label, _ = toa.tr.predict(frame)
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

    def get_block_info(self):
        with self.orm.session() as s:
            strk = s.query(Strike).filter_by(id=self.strike_db_id).first()
            blk = s.query(Block).filter_by(id=strk.block_id).first()
            return blk.__dict__


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
        self.check_arguments()
        self.strike_position = (self.payload.get('x'), self.payload.get('y'))

    def check_arguments(self):
        if self.loader is not None:
            if self.payload is None:
                self.payload = self.loader.info
            if self.pose_df is None:
                self.pose_df = self.loader.frames_df.copy()
                # smoothing of x and y
                for c in ['x', 'y']:
                    self.pose_df[f'orig_{c}'] = self.pose_df[c].copy()
                    self.pose_df[c] = savgol_filter(self.pose_df[c], window_length=37, polyorder=0)
            if self.bug_traj is None:
                self.bug_traj = self.loader.traj_df

        assert self.pose_df is not None and len(self.pose_df) > 1
        self.calc_kinematics()
        mandatory_fields = ['time', 'x', 'y', 'bug_x', 'bug_y', 'is_hit']
        for fld in mandatory_fields:
            assert fld in self.payload, f'Field {fld} was not found in strike payload'

    def calc_kinematics(self):
        for c in ['velocity', 'velocity_y', 'acceleration', 'acceleration_y']:
            self.pose_df[c] = np.nan

        t = self.pose_df.time.view('int64').values / 1e9
        t = t - t[self.time_zero_frame]
        self.pose_df['rel_time'] = t
        kdf = self.pose_df.loc[self.relevant_frames].copy()
        dt = float(np.mean(np.diff(kdf.rel_time)))

        vx, vy = self.calc_derivative(kdf.orig_x, kdf.orig_y, dt, deriv=1)
        self.pose_df.loc[self.relevant_frames, 'velocity_y'] = vy[1:]
        self.pose_df.loc[self.relevant_frames, 'velocity'] = np.sqrt(vy**2 + vx**2)[1:]
        ax, ay = self.calc_derivative(kdf.orig_x, kdf.orig_y, dt, deriv=2)
        self.pose_df.loc[self.relevant_frames, 'acceleration_y'] = ay[1:]
        self.pose_df.loc[self.relevant_frames, 'acceleration'] = np.sqrt(ax**2 + ay**2)[1:]

    def get_lizard_acceleration(self):
        af = self.pose_df.loc[self.leap_frame:self.strike_frame_id+1, 'acceleration']
        max_accl = af.min()
        max_accl_delay = (self.pose_df.loc[af.idxmin(), 'time'] - self.pose_df.loc[self.leap_frame, 'time']).total_seconds()
        return max_accl, max_accl_delay

    def get_bug_kinematics_before_leap(self, n_back=120, is_xy=True):
        traj = self.bug_traj.loc[self.bug_traj_leap_index-n_back:self.bug_traj_leap_index+1]
        d = dict()
        d['pos_x'], d['pos_y'] = traj[['x', 'y']].iloc[-1].values
        if is_xy:
            for c in ['x', 'y']:
                v = traj[c].diff() / traj.time.diff().dt.total_seconds()
                d[f'v{c}_mean'] = v.mean()
                d[f'v{c}_std'] = v.std()
                a = v.diff() / traj.time.diff().dt.total_seconds()
                d[f'a{c}_mean'] = a.mean()
                d[f'a{c}_std'] = a.std()
                jerk = a.diff() / traj.time.diff().dt.total_seconds()
                d[f'jerk{c}_mean'] = jerk.mean()
                d[f'jerk{c}_std'] = jerk.std()
        else:
            dist = np.sqrt(traj.x.diff() ** 2 + traj.y.diff() ** 2)
            v = dist / traj.time.diff().dt.total_seconds()
            d['v_mean'] = v.mean()
            d['v_std'] = v.std()
            a = v.diff() / traj.time.diff().dt.total_seconds()
            d['a_mean'] = a.mean()
            d['a_std'] = a.std()
        return d

    def plot_strike_analysis(self, n=6, only_save_to=None, only_return=False):
        # get tongue frame IDs
        fig = plt.figure(figsize=(20, 15))
        grid = fig.add_gridspec(ncols=n, nrows=3, width_ratios=[1]*n, height_ratios=[3, 3, 5], hspace=0.2)
        self.plot_frames_sequence(grid[:2, :], n)
        self.plot_kinematics(grid[2, :4])
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

    def plot_frames_sequence(self, grid, n, extra=30):
        inner_grid = grid.subgridspec(3, n, wspace=0.1, hspace=0.1)
        axes = inner_grid.subplots()  # Create all subplots for the inner grid.
        axes = axes.flatten()
        labels2track = ['strike_frame_id', 'calc_strike_frame', 'leap_frame']
        frames_ids = np.linspace((self.leap_frame or (self.strike_frame_id - 120)) - extra,
                                 self.strike_frame_id + extra, 3 * n).astype(int)
        step = max(np.diff(frames_ids))
        frames_iter = self.loader.gen_frames(frames_ids)
        for i, (frame_id, frame) in enumerate(frames_iter):
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            frame = PredPlotter.plot_single_part(self.loader.frames_df, frame_id, frame)
            h, w = frame.shape[:2]
            frame = frame[h//2:, 350:w-350]
            labels = []
            for lfi in labels2track:
                lfix = getattr(self, lfi)
                if lfix is None:
                    continue
                d = frame_id - lfix
                if np.abs(d) < step:
                    if d != 0:
                        lfi = f"{'+' if d > 0 else '-'}{lfi}"
                    labels.append(lfi)
            if frame_id in self.tongue_frame_ids:
                labels.append('tongue')
            if self.pose_df.loc[frame_id, 'is_in_screen'] >= 0:
                labels.append('in_screen')

            h_text, text_colors = 30, {'strike_frame_id': (255, 0, 0), 'calc_strike_frame': (255, 255*0.647, 0),
                                       'tongue': (0, 255, 0), 'leap_frame': (0, 0, 255), 'in_screen': (0, 0, 0)}
            for lbl in labels:
                lbl_ = lbl[1:] if lbl.startswith('+') or lbl.startswith('-') else lbl
                frame = cv2.putText(frame, lbl, (20, h_text), cv2.FONT_HERSHEY_PLAIN, 3,
                                    text_colors[lbl_], 2, cv2.LINE_AA)
                h_text += 30
            axes[i].imshow(frame)
            axes[i].set_xticks([]); axes[i].set_yticks([])
            axes[i].set_title(frame_id)
            # axes[i].set_title('\n'.join(labels))

    def plot_kinematics(self, grid):
        inner_grid = grid.subgridspec(1, 3, wspace=0.2, hspace=0.2)
        axes = inner_grid.subplots()
        rel_df = self.pose_df.loc[self.relevant_frames]
        for ax, (label, seg) in zip(axes, {'position': rel_df.y,
                                           'velocity': rel_df.velocity_y,
                                           'acceleration': rel_df.acceleration_y}.items()):
            t = rel_df.rel_time.values
            ax.plot(t, seg, color='k')
            self._plot_strikes_lines(ax, rel_df.rel_time.diff().mean())
            ax.set_title(f'Nose {label} vs. frame_ids')
            ax.set_xlabel('Time around strike [sec]')
            if label == 'position':
                ax.set_ylim([750, 950])
                ax.legend()
            elif label == 'acceleration':
                peaks_idx, _ = find_peaks(seg, height=50, distance=10)
                peaks_idx = [int(pk) for pk in peaks_idx if t[pk] < 0]
                max_peak_id = peaks_idx[np.argmax(seg.iloc[peaks_idx])]
                ax.scatter(t[max_peak_id], seg.iloc[max_peak_id])

    def _plot_strikes_lines(self, ax, dt):
        if self.leap_frame:
            ax.axvline(self.pose_df.loc[self.leap_frame, 'rel_time'], linestyle='--', color='blue',
                       label=f'leap_frame ({self.leap_frame})')
        if self.strike_frame_id in self.relevant_frames:
            ax.axvline(self.pose_df.loc[self.strike_frame_id, 'rel_time'], linestyle='--', color='r',
                       label=f'strike_frame ({self.strike_frame_id})')
        if self.calc_strike_frame in self.relevant_frames:
            ax.axvline(self.pose_df.loc[self.calc_strike_frame, 'rel_time'], linestyle='--', color='orange',
                       label=f'calc strike frame ({self.calc_strike_frame})')
        ymin, ymax = ax.get_ylim()
        for frame_id in self.relevant_frames:
            if frame_id in self.tongue_frame_ids:
                x = self.pose_df.loc[frame_id, 'rel_time']
                ax.add_patch(plt.Rectangle((x, ymin), dt, ymax-ymin, color='lightgreen', alpha=0.4))
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
    def relevant_frames(self):
        return np.arange(self.time_zero_frame - self.loader.n_frames_back,
                         self.time_zero_frame + self.loader.n_frames_forward + 1)

    @cached_property
    def proj_strike_pos(self):
        proj_strike, _ = self.get_projected_coords()
        return proj_strike

    @cached_property
    def proj_leap_pos(self):
        _, proj_leap = self.get_projected_coords()
        return proj_leap

    @cached_property
    def time_zero_frame(self):
        return self.strike_frame_id

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
    def bug_traj_leap_index(self):
        return closest_index(self.bug_traj.time, self.pose_df.loc[self.leap_frame, 'time'])

    @cached_property
    def tongue_frame_ids(self) -> list:
        tng = self.loader.frames_df[TONGUE_COL]
        return tng[tng == 1].index.tolist()

    @staticmethod
    def calc_derivative(x, y=None, dt=0.016, window_length=41, deriv=0):
        dx = savgol_filter(x, window_length=window_length, polyorder=2, deriv=deriv, delta=dt)
        if y is not None:
            dy = savgol_filter(y, window_length=window_length, polyorder=2, deriv=deriv, delta=dt)
            return np.insert(dx, 0, np.nan), np.insert(dy, 0, np.nan)
        else:
            return np.insert(dx, 0, np.nan)

    @cached_property
    def max_acceleration(self):
        rel_df = self.pose_df.loc[self.relevant_frames]
        accl = rel_df.acceleration_y.loc[self.leap_frame:self.strike_frame_id]
        return accl.max()

    @cached_property
    def bug_traj_last_frame(self):
        return closest_index(self.pose_df.time, self.bug_traj.time.iloc[-1])

    @cached_property
    def strike_frame_id(self):
        return closest_index(self.pose_df.time, self.payload.get('time'))

    @cached_property
    def leap_frame(self):
        # yf = self.pose_df[['time', 'y']]
        # yf = yf[~yf.y.isnull()]
        # t = yf.time.values.astype(np.int64) / 10 ** 9
        # yf.loc[yf.index[2:], 'accl'] = self.calc_derivative(self.calc_derivative(yf.y.values, t), t[1:])
        stop_frame_id = self.pose_df[~self.pose_df.velocity_y.isnull()].index[0]
        v = self.pose_df.loc[stop_frame_id:self.strike_frame_id-10, 'acceleration_y']
        cross_idx = v[np.sign(v).diff().fillna(0) == 2].index.tolist()
        if len(cross_idx) > 0:
            return cross_idx[-1]
        else:
            return None
        #
        # leap_frame_idx = None
        # grace_count = 0
        # for r in np.arange(self.strike_frame_id, stop_frame_id, -1):
        #     if self.pose_df.loc[r, 'acceleration'] > -0.05:
        #         if grace_count == 0:
        #             leap_frame_idx = r
        #         grace_count += 1
        #         if grace_count < 5:
        #             continue
        #         break
        #     else:
        #         grace_count = 0
        # return leap_frame_idx

    @cached_property
    def calc_strike_frame(self):
        max_diff_strike_frame = 150
        y = self.pose_df.y.values
        peaks_idx, _ = find_peaks(y, height=910, distance=10)
        peaks_idx = peaks_idx[(np.abs(peaks_idx - self.strike_frame_id) < max_diff_strike_frame) &
                              (peaks_idx < self.bug_traj_last_frame)]

        strike_frame_idx = self.strike_frame_id
        try:
            if len(peaks_idx) > 0:
                strike_frame_idx = peaks_idx[np.argmax(y[peaks_idx])]
            else:
                strike_ids = np.arange(self.strike_frame_id - max_diff_strike_frame // 2,
                                       self.strike_frame_id + max_diff_strike_frame // 2)
                around_strike_y_values = np.array([y[idx] for idx in strike_ids if idx in np.arange(len(y)).astype(int)])
                if len(around_strike_y_values) > 0:
                    strike_frame_idx = strike_ids[np.argmax(y[strike_ids])]
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
        if not traj_id:
            return
        bug_pos = self.bug_traj.loc[traj_id, ['x', 'y']].values.tolist()
        d = distance(*bug_pos, *self.strike_position)
        return pixels2cm(d)

    @cached_property
    def bug_speed(self):
        d = self.loader.bug_traj_before_strike.iloc[1:, :]
        dt = d.time.diff().dt.total_seconds().mean()
        vx, vy = self.calc_derivative(d.x, d.y, dt, deriv=1)
        v = np.sqrt(vx ** 2 + vy ** 2)
        # v = remove_outliers(np.sqrt(d.x ** 2 + d.y ** 2) / d.time.dt.total_seconds())
        return pixels2cm(np.nanmean(v))  # speed in cm/sec

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
        self.df = self.load_data()

    def load_data(self) -> pd.DataFrame:
        df = []
        sids = self.load_strikes()
        cache_df = pd.read_parquet(self.cache_path)
        for sid in tqdm(sids, desc='loading data'):
            if sid < 6546:
                continue
            if self.is_use_cache and sid in cache_df.strike_id.values:
                df.append(cache_df.query(f'strike_id=={sid}').iloc[0].to_dict())
            else:
                try:
                    ld = Loader(sid, 'front', is_debug=False, is_load_tongue=False)
                    sa = StrikeAnalyzer(ld)
                    bug_pos = sa.get_bug_pos_at_frame(sa.calc_strike_frame).flatten()
                    strike_pos = sa.payload['x'], sa.payload['y']
                    df.append({'strike_id': sid, 'time': ld.info.get('time'), 'bug_pos_x': bug_pos[0], 'bug_pos_y': bug_pos[1],
                               'strike_pos_x': strike_pos[0], 'strike_pos_y': strike_pos[1]})
                except MissingStrikeData:
                    continue
                except Exception as exc:
                    print(f'Error strike-{sid}: {exc}')
        df = pd.DataFrame(df)
        df['miss'] = distance(df.strike_pos_x, df.strike_pos_y, df.bug_pos_x, df.bug_pos_y)
        df.to_parquet(self.cache_path)
        return df

    @property
    def cache_path(self):
        return Path('/data/Pogona_Pursuit/output/strike_analysis/multi_trial') / 'circle.parquet'

    def plot_circle_strikes(self):
        df_ = self.df.query('"2022-12-08" <= time < "2022-12-09" and bug_pos_x < 1300 and miss < 300')
        bug_x, bug_y = df_.bug_pos_x.values, df_.bug_pos_y.values
        strk_x, strk_y = df_.strike_pos_x.values, df_.strike_pos_y.values
        xc, yc, R = fit_circle(bug_x, bug_y)
        # idx = np.where((distance(bug_x, bug_y, xc, yc) < 1.1 * R) & (distance(strk_x, strk_y, xc, yc) < 1.6 * R))[0]
        ax = plt.subplot()

        ax.scatter(bug_x, bug_y, c='g', label='bug position')
        ax.scatter(strk_x, strk_y, c='r', label='hit position')
        for i, row in df_.iterrows():
            ax.plot([row.strike_pos_x, row.bug_pos_x], [row.strike_pos_y, row.bug_pos_y], color='goldenrod', linewidth=0.5)
        ax.add_patch(plt.Circle((xc, yc), R, color='k', fill=False))
        ax.invert_yaxis()
        ax.set_xlabel('Monitor-X [pixels]')
        ax.set_ylabel('Monitor-Y [pixels]')
        ax.legend()
        ax.set_title('PV67')
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


def load_strikes(animal_id, start_time=None, movement_type=None):
    orm = ORM()
    strikes_ids = []
    with orm.session() as s:
        exps = s.query(Experiment).filter_by(animal_id=animal_id)
        if start_time:
            exps = exps.filter(Experiment.start_time >= start_time)
        for exp in exps.all():
            for blk in exp.blocks:
                if movement_type and blk.movement_type != movement_type:
                    continue
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


def get_pose_cache_file(video_path):
    preds_dir = Path(video_path).parent / 'predictions'
    preds_dir.mkdir(exist_ok=True)
    vid_name = Path(video_path).with_suffix('.parquet').name
    return preds_dir / f'short_{POSE_MODEL_NAME}_{vid_name}'


def short_predict(animal_id, cam_name='front', n_frames_back=50, start_time=None, strike_id=None, movement_type=None):
    if strike_id:
        strikes_ids = [strike_id]
    else:
        strikes_ids = load_strikes(animal_id, start_time=start_time, movement_type=movement_type)
        strikes_ids = sorted(strikes_ids)
    dlc = DLCPose('front')
    vids = {}

    orm0 = ORM()
    for sid in strikes_ids:
        try:
            ld = Loader(sid, cam_name, is_debug=False, is_load_pose=False, is_load_tongue=False, orm=orm0)
            video_path = ld.video_path.as_posix()
            cache_path = get_pose_cache_file(video_path)

            if cache_path.exists():
                continue

            frames_range = (ld.strike_frame_id - ld.n_frames_back, ld.strike_frame_id + ld.n_frames_forward)
            vids.setdefault(video_path, []).append(frames_range)
        except Exception as exc:
            print(f'Error in loading Strike-{sid}; {exc}')

    with open('/data/Pogona_Pursuit/output/strike_analysis/strikes_vids.pickle', 'wb') as f:
        pickle.dump(vids, f)

    for video_path, frames_ranges in vids.items():
        try:
            pfs = []
            for start_frame, end_frame in frames_ranges:
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                for i in tqdm(range(start_frame, end_frame + 1), desc=f'Vid={video_path.split("/")[-1]}'):
                    ret, frame = cap.read()
                    pf = dlc._predict(frame, frame_id=i)
                    pfs.append(pf)
            pfs = pd.concat(pfs)
            pfs.to_parquet(get_pose_cache_file(video_path))
        except Exception as exc:
            print(f'Error analyzing video: {video_path}; {exc}')


def analyze_strikes(animal_id, cam_name='front', n_frames_back=50, start_time=None, strike_id=None, movement_type=None):
    if strike_id:
        strikes_ids = [strike_id]
    else:
        strikes_ids = load_strikes(animal_id, start_time=start_time, movement_type=movement_type)
        strikes_ids = sorted(strikes_ids)
    orm = ORM()
    for sid in tqdm(strikes_ids, desc='loading strikes'):
        try:
            ld = Loader(sid, cam_name, is_debug=False)
            # root_dir = Path('/data/Pogona_Pursuit/output/strike_analysis/PV80')
            # strk_dir = root_dir / str(sid)
            # strk_dir.mkdir(exist_ok=True)
            # for frame_id, frame in ld.gen_frames_around_strike(n_frames_back=180):
            #     cv2.imwrite((strk_dir / f'{frame_id}.jpg').as_posix(), frame)
            # ld.play_strike()
            # ld.plot_strike_events()
            sa = StrikeAnalyzer(ld)
            # df.append({'speed': sa.bug_speed, 'prediction_distance': sa.prediction_distance})

            sa.plot_strike_analysis(only_save_to='/data/Pogona_Pursuit/output/strike_analysis/PV80')
            with orm.session() as s:
                strk = s.query(Strike).filter_by(id=sid).first()
                strk.prediction_distance = sa.prediction_distance
                strk.calc_speed = sa.bug_speed
                strk.projected_strike_coords = sa.proj_strike_pos.tolist()
                strk.projected_leap_coords = sa.proj_leap_pos.tolist()
                strk.max_acceleration = sa.max_acceleration
                s.commit()
        except MissingStrikeData:
            print(f'Strike-{sid}: No timestamps for frames')
        except Exception as exc:
            print(f'Strike-{sid}: {exc}')

    # df = pd.DataFrame(df)
    # sns.scatterplot(data=df, x='speed', y='prediction_distance')
    # plt.show()


if __name__ == '__main__':
    # CircleMultiTrialAnalysis().plot_circle_strikes()
    # ld = Loader(5968, 'front')
    # sa = StrikeAnalyzer(ld)
    # sa.plot_strike_analysis()
    # delete_duplicate_strikes('PV80')
    # play_strikes('PV80', start_time='2022-12-01', cam_name='front', is_load_pose=False, strikes_ids=[6365])
    analyze_strikes('PV80', start_time='2023-01-29')#, movement_type='random')
    # short_predict('PV80')
    # foo()
    # calibrate()
    # save_strikes_dataset('/data/Pogona_Pursuit/output/datasets/pogona_tongue/', 'PV80')
