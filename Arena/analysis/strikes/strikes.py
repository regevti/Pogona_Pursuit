import os
import cv2
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from functools import cached_property
from scipy.signal import find_peaks, savgol_filter
if __name__ == '__main__':
    os.chdir('../..')

import config
from analysis.pose_utils import closest_index, pixels2cm, distance, fit_circle
from analysis.strikes.loader import Loader, MissingStrikeData
from db_models import ORM, Experiment, Block, Strike

NUM_FRAMES_TO_PLOT = 5
NUM_POSE_FRAMES_PER_STRIKE = 30


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
                df = self.pose_df[['time', 'angle']].copy()
                df.columns = df.columns.droplevel(1)
                self.pose_df = pd.concat([df, self.pose_df.nose.copy()], axis=1)
                # smoothing of x and y
                for c in ['x', 'y']:
                    self.pose_df[f'orig_{c}'] = self.pose_df[c].copy()
                    self.pose_df[c] = savgol_filter(self.pose_df[c], window_length=37, polyorder=0, mode='nearest')
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
        zero_frame_id = self.pose_df.index.to_list().index(self.time_zero_frame)
        t = t - t[zero_frame_id]
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
            # frame = self.loader.dlc_pose.predictor.plot_single_part(self.loader.frames_df, frame_id, frame)
            frame = self.loader.dlc_pose.predictor.plot_predictions(frame, frame_id, self.loader.frames_df)
            if frame is None:
                continue
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
            # if frame_id in self.tongue_frame_ids:
            #     labels.append('tongue')
            # if self.pose_df.loc[frame_id, 'is_in_screen'] >= 0:
            #     labels.append('in_screen')

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
                # ax.set_ylim([750, 950])
                ax.legend()
            elif label == 'acceleration':
                peaks_idx, _ = find_peaks(seg, height=50, distance=10)
                peaks_idx = [int(pk) for pk in peaks_idx if t[pk] < 0]
                if peaks_idx:
                    max_peak_id = peaks_idx[np.argmax(seg.iloc[peaks_idx])]
                    ax.scatter(t[max_peak_id], seg.iloc[max_peak_id])

    def _plot_strikes_lines(self, ax, dt):
        if self.leap_frame:
            ax.axvline(self.pose_df['rel_time'].loc[self.leap_frame], linestyle='--', color='blue',
                       label=f'leap_frame ({self.leap_frame})')
        if self.strike_frame_id in self.relevant_frames:
            ax.axvline(self.pose_df['rel_time'].loc[self.strike_frame_id], linestyle='--', color='r',
                       label=f'strike_frame ({self.strike_frame_id})')
        if self.calc_strike_frame in self.relevant_frames:
            ax.axvline(self.pose_df['rel_time'].loc[self.calc_strike_frame], linestyle='--', color='orange',
                       label=f'calc strike frame ({self.calc_strike_frame})')
        ymin, ymax = ax.get_ylim()
        # for frame_id in self.relevant_frames:
        #     if frame_id in self.tongue_frame_ids:
        #         x = self.pose_df['rel_time'].loc[frame_id]
        #         ax.add_patch(plt.Rectangle((x, ymin), dt, ymax-ymin, color='lightgreen', alpha=0.4))
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
        frames_ids = np.arange(self.time_zero_frame - self.loader.n_frames_back,
                         self.time_zero_frame + self.loader.n_frames_forward + 1)
        return [i for i in frames_ids if i in self.pose_df.index]

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
        bug_pos = self.get_bug_pos_at_frame(self.calc_strike_frame)
        if bug_pos is None:
            return None, None
        bug_pos = bug_pos.flatten()
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

    # @cached_property
    # def tongue_frame_ids(self) -> list:
    #     tng = self.loader.frames_df[TONGUE_COL]
    #     return tng[tng == 1].index.tolist()

    @staticmethod
    def calc_derivative(x, y=None, dt=0.016, window_length=41, deriv=0):
        dx = savgol_filter(x, window_length=window_length, polyorder=2, deriv=deriv, delta=dt, mode='nearest')
        if y is not None:
            dy = savgol_filter(y, window_length=window_length, polyorder=2, deriv=deriv, delta=dt, mode='nearest')
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
        try:
            stop_frame_id = self.pose_df[~self.pose_df.velocity_y.isnull()].index[0]
            v = self.pose_df.loc[stop_frame_id:self.strike_frame_id-10, 'acceleration_y']
            cross_idx = v[np.sign(v).diff().fillna(0) == 2].index.tolist()
            if len(cross_idx) > 0:
                return cross_idx[-1]
        except Exception:
            pass
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


def play_strikes(animal_id, start_time=None, cam_name='front', is_load_pose=True, strikes_ids=None):
    if not strikes_ids:
        strikes_ids = load_strikes(animal_id, start_time)
    for sid in strikes_ids:
        try:
            ld = Loader(sid, cam_name, is_load_pose=is_load_pose)
            ld.play_strike()
        except Exception as exc:
            print(f'ERROR strike_id={sid}: {exc}')


class StrikeScanner:
    def __init__(self, logger=None, animal_id=None, cam_name='front', is_skip_committed=True,
                 start_time=None, strike_id=None, movement_type=None, is_plot_summary=False):
        """ Scan and analyze strikes.
        @param logger: external logger to be used. If none this class uses the print function
        @param animal_id: scan only for specified animal_id. If none, scan all animals.
        @param cam_name: cam to use for analysis. Default is front
        @param is_skip_committed: skip already analyzed strikes.
        @param start_time: The start time for the strikes scan. If none, scan all.
        @param strike_id: Limit the scan to specified strikes IDs.
        @param movement_type: Use only strikes from blocks with a specific movement type. If none, use all.
        @param is_plot_summary: Plot a pdf summary of the analysis to output_dir
        """
        self.animal_id = animal_id
        self.cam_name = cam_name
        self.logger = logger
        self.start_time = start_time
        self.strike_id = strike_id
        self.movement_type = movement_type
        self.is_plot_summary = is_plot_summary
        self.is_skip_committed = is_skip_committed
        self.orm = ORM()
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def scan(self):
        strikes_ids = self.load_strikes_ids()
        if not strikes_ids:
            return
        n_committed = 0
        errors = []
        for sid in tqdm(strikes_ids, desc='Strikes Scan'):
            try:
                ld = Loader(sid, self.cam_name, is_debug=False, orm=self.orm)
                sa = StrikeAnalyzer(ld)
                if self.is_plot_summary:
                    sa.plot_strike_analysis(only_save_to=self.output_dir.as_posix())
                self.commit_strike_analysis(sid, sa)
                n_committed += 1

            except MissingStrikeData as exc:
                self.commit_analysis_error(sid, str(exc))
                errors.append(f'Strike-{sid}: Missing strike data; {exc}')
            except Exception as exc:
                self.commit_analysis_error(sid, str(exc))
                errors.append(f'Strike-{sid}: {exc}\n{traceback.format_exc()}\n')

        if errors:
            errors = "\n".join(errors)
            self.print(f'Errors in the analysis:\n{errors}')
        if n_committed:
            self.print(f'Finished analyze_strikes; analyzed and committed {n_committed} strikes')

    def load_strikes_ids(self):
        strikes_ids = []
        with self.orm.session() as s:
            exps = s.query(Experiment)
            if self.animal_id:
                exps = exps.filter_by(animal_id=self.animal_id)
            if self.start_time:
                exps = exps.filter(Experiment.start_time >= self.start_time)
            for exp in exps.all():
                for blk in exp.blocks:
                    if self.movement_type and blk.movement_type != self.movement_type:
                        continue
                    for strk in blk.strikes:
                        if strk.is_climbing or (self.is_skip_committed and (strk.calc_speed or strk.analysis_error)):
                            continue
                        strikes_ids.append(strk.id)
        return sorted(strikes_ids)

    def commit_strike_analysis(self, sid: int, sa: StrikeAnalyzer):
        """commit strike analysis to Strike table in the DB"""
        with self.orm.session() as s:
            strk = s.query(Strike).filter_by(id=sid).first()
            strk.prediction_distance = sa.prediction_distance
            strk.calc_speed = sa.bug_speed
            if sa.proj_strike_pos is not None:
                strk.projected_strike_coords = sa.proj_strike_pos.tolist()
            if sa.proj_leap_pos is not None:
                strk.projected_leap_coords = sa.proj_leap_pos.tolist()
            strk.max_acceleration = sa.max_acceleration
            s.commit()

    def commit_analysis_error(self, sid, error_msg):
        with self.orm.session() as s:
            strk = s.query(Strike).filter_by(id=sid).first()
            strk.analysis_error = error_msg
            s.commit()

    def print(self, msg):
        print_func = print if self.logger is None else self.logger.info
        print_func(msg)

    @property
    def output_dir(self):
        return Path(f'{config.OUTPUT_DIR}/strike_analysis/{self.animal_id}')


def extract_bad_annotated_strike_frames(animal_id, cam_name='front', strike_id=None, n=6, extra=30, **kwargs):
    if strike_id:
        strikes_ids = [strike_id]
    else:
        strikes_ids = load_strikes(animal_id, **kwargs)
        strikes_ids = sorted(strikes_ids)

    output_dir = '/data/Pogona_Pursuit/output/models/deeplabcut/train/front_head_only/labeled-data/march23'
    for sid in tqdm(strikes_ids, desc='loading strikes'):
        try:
            ld = Loader(sid, cam_name, is_debug=False)
            sa = StrikeAnalyzer(ld)
            frames_ids = np.linspace((sa.leap_frame or (sa.strike_frame_id - 120)) - extra,
                                     sa.strike_frame_id + extra, 3 * n).astype(int)
            step = max(np.diff(frames_ids))
            frames_iter = ld.gen_frames(frames_ids)
            for i, (frame_id, frame) in enumerate(frames_iter):
                if any([ld.frames_df[bp].loc[frame_id, 'prob'] < 0.5 for bp in ld.dlc_pose.predictor.bodyparts]):
                    cv2.imwrite(f'{output_dir}/{ld}_{frame_id}.jpg', frame)
        except MissingStrikeData as exc:
            print(f'Strike-{sid}: No timestamps for frames; {exc}')
        except Exception as exc:
            print(f'Strike-{sid}: {exc}\n{traceback.format_exc()}\n')


if __name__ == '__main__':
    # CircleMultiTrialAnalysis().plot_circle_strikes()
    # ld = Loader(5968, 'front')
    # sa = StrikeAnalyzer(ld)
    # sa.plot_strike_analysis()
    # delete_duplicate_strikes('PV80')
    # play_strikes('PV80', start_time='2022-12-01', cam_name='front', is_load_pose=False, strikes_ids=[6365])
    StrikeScanner(is_skip_committed=False).scan()
    # extract_bad_annotated_strike_frames('PV85')#, movement_type='random')
    # short_predict('PV80')
    # foo()
    # calibrate()
    # save_strikes_dataset('/data/Pogona_Pursuit/output/datasets/pogona_tongue/', 'PV80')
