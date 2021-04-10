from pathlib import Path
import numpy as np
from functools import lru_cache
import pandas as pd
import pickle
import cv2
from pose import PoseAnalyzer
from scipy.signal import find_peaks
from loader import Loader, closest_index
from pose_utils import *


NUM_FRAMES_TO_PLOT = 5
NUM_POSE_FRAMES_PER_STRIKE = 30


class TrialStrikes:
    def __init__(self, loader: Loader):
        self.loader = loader
        self.hits_df = loader.hits_df
        self.pose_df = self.get_pose()
        self.strikes = [StrikeSummary(loader, i, self.pose_df) for i in self.hits_df.index]

    def strikes_summary(self, is_plot=True, is_close_plot=False, use_cache=True):
        """
        Main function for strikes analysis
        :param is_plot - show the figure or not
        :param is_close_plot - close figure after it was created and saved
        :param use_cache - Use cached files if exist, instead of running analysis. If not exist, run anyway.
        """
        if use_cache:
            return [s.load(is_plot) for s in self.strikes]

        print(f'>> Start strikes summary for {self.loader}')
        frames, frames_ids = self.load_trial_frames(NUM_FRAMES_TO_PLOT)
        all_data = []
        for i, frame_group in enumerate(frames_ids):
            if not frame_group:
                print(f'Unable to find frames for strike #{i}')
                continue
            data = self.strikes[i].summary_plot(frame_group, frames, is_close_plot)
            all_data.append(data)
        return all_data

    def load_trial_frames(self, n_frames):
        d = (n_frames - 1) // 2
        frames_ids = [list(range(f - d, f + d + 1)) if f else [] for f in self.loader.get_hits_frames()]
        flat_frames = [f for sublist in frames_ids for f in sublist]
        if not flat_frames:
            msg = 'No Frames were found; '
            for i, hit in self.hits_df.iterrows():
                msg += f'hit #{i + 1} time: {hit.get("time").tz_convert("utc").tz_localize(None)} not in ' \
                       f'{self.loader.frames_ts[0]} - {self.loader.frames_ts[self.loader.frames_ts.index[-1]]}\n'
            raise Exception(msg)
        cap = cv2.VideoCapture(self.loader.video_path.as_posix())
        frames = {}
        for frame_id in range(max(flat_frames) + 1):
            ret, frame = cap.read()
            if frame_id in flat_frames:
                frames[frame_id] = frame
        cap.release()
        return frames, frames_ids

    def get_pose(self):
        """Load the pose csv using pose.Analyzer"""
        a = PoseAnalyzer(self.loader)
        try:
            return a.run_pose(load_only=True)
        except Exception:
            return


class StrikeSummary:
    def __init__(self, loader: Loader, strike_idx: int, pose_df: pd.DataFrame):
        self.loader = loader
        self.strike_idx = strike_idx
        self.hit_df = loader.hits_df.loc[strike_idx, :]
        hits_frames = self.loader.get_hits_frames()
        self.strike_frame = hits_frames[strike_idx] if strike_idx < len(hits_frames) else None
        self.pose_df = pose_df
        self.strike_errors = []

    def summary_plot(self, frame_group, frames, is_close_plot=False):
        fig = plt.figure(figsize=(20, 15))
        grid = fig.add_gridspec(ncols=4, nrows=3, width_ratios=[1, 1, 1, 1], height_ratios=[5, 6, 5])
        fig.suptitle(f'Results for strike #{self.strike_idx + 1} ({"hit" if self.is_hit else "miss"})\n{self.loader}',
                     fontsize=15)
        self.plot_frames(grid[0, :], frame_group, frames)
        if self.bug_traj is not None:
            self.plot_bug_trajectory(fig.add_subplot(grid[1, 0]))
            self.plot_projected_strike(fig.add_subplot(grid[1, 1]))
        if self.nose_df is not None:
            self.plot_xy_pose(fig.add_subplot(grid[1, 2]))
            self.plot_nose_vs_time(fig.add_subplot(grid[1, 3]))
        self.plot_info(fig.add_subplot(grid[2, 0]))
        self.plot_errors(fig.add_subplot(grid[2, 1:]))
        fig.tight_layout()
        fig.subplots_adjust(top=0.95, hspace=0.05)
        fig.patch.set_linewidth(3)
        fig.patch.set_edgecolor('black')

        self.saved_image_folder.mkdir(exist_ok=True)
        self.save()
        fig.savefig(self.save_image_path)
        if is_close_plot:
            plt.close(fig)

        return self.data

    def plot_frames(self, grid, frame_group, frames):
        """Plot the frames close to the strike"""
        cols = 5
        rows = int(np.ceil(len(frame_group) / cols))
        inner_grid = grid.subgridspec(rows, cols, wspace=0.1, hspace=0)
        axes = inner_grid.subplots()  # Create all subplots for the inner grid.
        axes = axes.flatten()
        for i, f_id in enumerate(frame_group):
            if frames.get(f_id) is None:
                continue
            axes[i].imshow(frames[f_id])
            if f_id == self.strike_frame:
                axes[i].set_title('strike frame')

    def plot_bug_trajectory(self, ax):
        try:
            cl = colorline(ax, self.bug_traj.x.to_numpy(), self.bug_traj.y.to_numpy(), alpha=1)
            plt.colorbar(cl, ax=ax, orientation='horizontal')
            ax.scatter(self.hit_df.x, self.hit_df.y, marker='X', color='r', label='strike', zorder=3)
            ax.add_patch(plt.Circle((self.hit_df.bug_x, self.hit_df.bug_y),
                                    self.bug_radius, color='lemonchiffon', alpha=0.4))
            if self.leap_frame:
                leap_bug_traj = self.loader.bug_data_for_frame(self.leap_frame)
                if leap_bug_traj is not None:
                    ax.scatter(leap_bug_traj.x, leap_bug_traj.y, color='g', marker='D', s=50, label='bug at leap start',
                               zorder=2)
            ax.set_xlim([0, 2400])
            ax.set_ylim([-60, 1000])
            ax.set_title(f'Bug Trajectory\ncalculated speed: {self.bug_speed:.1f} cm/sec')
            ax.invert_yaxis()
            ax.legend()
        except Exception as exc:
            self.log(f'Error plotting bug trajectory; {exc}')

    def plot_projected_strike(self, ax, default_leap=20, is_plot_strike_only=False):
        try:
            if self.leap_frame:
                leap_pos = self.loader.bug_data_for_frame(self.leap_frame)[['x', 'y']].to_numpy(dtype='float')
            else:
                leap_pos = self.bug_traj.loc[self.bug_traj.index[-default_leap], ['x', 'y']].to_numpy(dtype='float')
            strike_pos, proj_leap_pos = self.project_strike_coordinates(leap_pos)
            ax.scatter(*strike_pos, label='strike', color='r', marker='X')
            if is_plot_strike_only:
                return
            ax.scatter(0, 0, label='bug position at strike', color='m', marker='D')
            if self.leap_frame:
                ax.scatter(*proj_leap_pos, label='bug position at leap', color='g', marker='D')
            ax.add_patch(plt.Circle((0, 0), self.bug_radius, color='lemonchiffon', alpha=0.4))
            ax.legend()
            lim = max([*np.abs(strike_pos), *np.abs(proj_leap_pos), self.bug_radius])
            ax.set_xlim([-lim, lim])
            ax.set_ylim([-lim, lim])
            ax.plot([0, 0], [-lim, lim], 'k')
            ax.plot([-lim, lim], [0, 0], 'k')
            title = 'Projected Strike Coordinates'
            if pd:
                title += f'\nPD: {pd:.2f} cm'
            ax.set_title(title)
        except Exception as exc:
            self.log(f'Error plotting projected strike; {exc}')

    def plot_xy_pose(self, ax):
        cl = colorline(ax, self.nose_df.x.to_numpy(), self.nose_df.y.to_numpy())
        plt.colorbar(cl, ax=ax, orientation='horizontal')
        ax.scatter(self.nose_df.x[self.strike_frame], self.nose_df.y[self.strike_frame],
                   marker='X', color='r', label='strike', zorder=3)
        ax.legend()
        ax.set_title('XY of nose position')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    def plot_nose_vs_time(self, ax):
        title = 'Y of nose vs. frame'
        ax.plot(self.nose_df.y, label='nose')
        ax.scatter(self.strike_frame, self.nose_df.y[self.strike_frame], marker='X', color='r', label='logged_strike')
        if self.calc_strike_frame is not None:
            ax.scatter(self.calc_strike_frame, self.nose_df.y[self.calc_strike_frame], marker='X', color='y', label='calc_strike')
            title += f'\nY(strike) = {self.nose_df.y[self.calc_strike_frame]:.2f}'
        if self.leap_frame is not None:
            ax.scatter(self.leap_frame, self.nose_df.y[self.leap_frame], marker='D', color='g', label='leap')
        if self.leap_frame is not None and self.calc_strike_frame is not None:
            title += f'\nLeap Frame Diff: {self.calc_strike_frame - self.leap_frame}'

        ax.legend()
        ax.set_title(title)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Y (nose)')

    def plot_info(self, ax, w=30):
        ax.axis('off')
        text = f'Info:\n\n'
        for k, v in self.loader.info.items():
            if k.lower().startswith('block'):
                continue
            if isinstance(v, str) and len(v) > w:
                v_new = ''
                for i in range(0, len(v), w):
                    if i + w < len(v):
                        v_new += f'{v[i:i+w]}-\n'
                    else:
                        v_new += f'{v[i:]}'
                v = v_new
            text += f'{k}: {v}\n'
        ax.text(0, 0, text, fontsize=14, wrap=True)

    def plot_errors(self, ax):
        ax.axis('off')
        if not self.strike_errors:
            return
        text = 'Errors:\n\n'
        text += '\n'.join(self.strike_errors)
        ax.text(0, 0, text, fontsize=14)

    def project_strike_coordinates(self, leap_pos):
        """Project bug trajectory space to the vector spanned from bug position at leap to bug position at strike"""
        bug_pos = np.array([self.hit_df.bug_x, self.hit_df.bug_y])
        strike_pos = np.array([self.hit_df.x, self.hit_df.y])
        u = bug_pos - leap_pos
        xr = np.array([1, 0])
        th = np.arctan2(np.linalg.det([u, xr]), np.dot(u, xr))
        r = np.array(((np.cos(th), -np.sin(th)),
                      (np.sin(th), np.cos(th))))
        r0 = r.dot(bug_pos)
        return r.dot(strike_pos) - r0, r.dot(leap_pos) - r0

    @property
    @lru_cache()
    def leap_frame(self):
        if self.nose_df is None or len(self.nose_df) < 1:
            return
        y = self.nose_df.y
        dy = y.diff()
        leap_frame_idx = y.index[0]
        grace_count = 0
        for r in np.arange(self.calc_strike_frame, dy.index[0], -1):
            if dy[r] < 0.9:
                if grace_count == 0:
                    leap_frame_idx = r
                grace_count += 1
                if grace_count < 5:
                    continue
                break
            else:
                grace_count = 0
        # if y[self.calc_strike_frame] - y[leap_frame_idx] < 10:
        #     return
        return leap_frame_idx

    @property
    @lru_cache()
    def calc_strike_frame(self):
        strike_frame_idx = self.strike_frame
        if self.nose_df is None:
            return strike_frame_idx
        y = self.nose_df.y
        max_diff_strike_frame = 8
        peaks, _ = find_peaks(y.to_numpy(), height=870, distance=10)
        peaks_idx = y.index[peaks]
        peaks_idx = peaks_idx[np.abs(peaks_idx - self.strike_frame) < max_diff_strike_frame]
        try:
            if len(peaks_idx) > 0:
                strike_frame_idx = y[peaks_idx].idxmax()
            else:
                yrange = np.arange(self.strike_frame - max_diff_strike_frame, self.strike_frame + max_diff_strike_frame)
                yrange = [idx for idx in yrange if idx in y.dropna().index]
                if len(yrange) > 0:
                    strike_frame_idx = y[yrange].idxmax()
        except Exception as exc:
            print(f'Error in calc_strike_frame: {exc}')
        return strike_frame_idx

    @property
    @lru_cache()
    def pd(self):
        if self.leap_frame is None:
            return
        leap_bug_traj = self.loader.bug_data_for_frame(self.leap_frame)
        if leap_bug_traj is None:
            return
        d = distance(leap_bug_traj.x, leap_bug_traj.y, self.hit_df.x, self.hit_df.y)
        return pixels2cm(d)

    @property
    @lru_cache()
    def nose_df(self):
        if self.pose_df is None or self.strike_frame is None:
            return
        n_pose_frames = 2
        selected_frames = np.arange(self.strike_frame - n_pose_frames, self.strike_frame + n_pose_frames)
        selected_frames = [idx for idx in selected_frames if idx in self.pose_df.nose.index]
        return self.pose_df.nose.loc[selected_frames, :].copy()

    @property
    @lru_cache()
    def bug_speed(self):
        if self.bug_traj is None:
            return self.loader.calc_speed
        d = self.bug_traj.diff().iloc[1:, :]
        v = remove_outliers(np.sqrt(d.x ** 2 + d.y ** 2) / d.time.dt.total_seconds())
        return pixels2cm(v.mean())  # speed in cm/sec

    @property
    @lru_cache()
    def bug_traj(self):
        try:
            return self.loader.get_bug_trajectory_before_strike(self.strike_idx, n_records=120, max_dist=0.4)
        except Exception as exc:
            pass
            # self.log(f'Error loading bug traj: {exc}')

    @property
    @lru_cache()
    def bug_type(self):
        return self.hit_df.bug_type

    @property
    @lru_cache()
    def speed_group(self):
        if 2 <= self.bug_speed < 6:
            return 4
        elif 6 <= self.bug_speed < 10:
            return 8
        elif 10 <= self.bug_speed < 14:
            return 12

    @property
    @lru_cache()
    def bug_radius(self):
        bugs = {
            'cockroach': 150,
            'worm': 225,
            'red_beetle': 165,
            'black_beetle': 160,
            'green_beetle': 155
        }
        bug_type = self.hit_df.bug_type
        return bugs.get(bug_type, 160) / 1.5

    @property
    def is_hit(self):
        return self.hit_df.is_hit

    @property
    @lru_cache()
    def data(self):
        data_vars = ['bug_traj', 'strike_frame', 'leap_frame', 'calc_strike_frame', 'is_hit', 'bug_type',
                     'bug_radius', 'pd', 'bug_speed', 'speed_group', 'pose_df', 'bug_type', 'save_image_path']
        return {k: getattr(self, k, None) for k in data_vars}

    def load(self, is_plot):
        try:
            if is_plot:
                plt.imshow(cv2.imread(self.save_image_path.as_posix()))
            with self.save_pickle_path.open('rb') as f:
                data = pickle.load(f)
        except Exception as exc:
            self.log(f'Error loading cached summary for strike {self.strike_idx + 1}; {exc}')
            return {}

        return data

    def save(self):
        with self.save_pickle_path.open('wb') as f:
            pickle.dump(self.data, f)

    def log(self, msg):
        print(msg)
        self.strike_errors.append(msg)

    @property
    def save_image_path(self):
        return self.saved_image_folder / f'strike{self.strike_idx + 1}.jpg'

    @property
    def save_pickle_path(self):
        return self.saved_image_folder / f'strike{self.strike_idx + 1}.pickle'

    @property
    def saved_image_folder(self):
        return self.loader.trial_path / 'strike_analysis'
