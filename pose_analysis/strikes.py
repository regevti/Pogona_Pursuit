from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import numpy as np
import pandas as pd
import pickle
import cv2
from pose import Analyzer, BODY_PARTS
from scipy.signal import find_peaks
from loader import Loader, closest_index

NUM_FRAMES_BACK = 200


class StrikesSummary:
    def __init__(self, loader):
        self.loader = loader
        self.hits_df = loader.hits_df
        self.pose_df = self.get_pose()
        self.strike_errors = None

    def strikes_summary(self, n_frames=5, n_pose_frames=30, is_plot=True, is_close_plot=False, use_cache=True):
        """
        Main function for strikes analysis
        :param n_frames - Number of frames to show around the strike frame
        :param n_pose_frames - Number of frames (before&after) to be used in pose plots
        :param is_plot - show the figure or not
        :param is_close_plot - close figure after it was created and saved
        :param use_cache - Use cached files if exist, instead of running analysis. If not exist, run anyway.
        """
        assert n_frames % 2, 'n_frames must be odd number'
        self.strike_errors = []
        if use_cache:
            data, images_paths = self.load_cached_summary(is_plot)
            return data, images_paths

        print(f'>> Start strikes summary for {self.loader}')
        frames, frames_ids = self.load_trial_frames(n_frames)
        data = []
        images_paths = []
        for i, frame_group in enumerate(frames_ids):
            if not frame_group:
                print(f'Unable to find frames for strike #{i}')
                continue
            nose = None
            leap_frame_id = None
            calc_strike_frame = None
            pd = None
            try:
                is_hit = self.hits_df.loc[i, "is_hit"]
                strike_frame = int(np.median(frame_group))
                bug_traj = self.get_bug_trajectory_before_strike(i)
                bug_speed = self.calculate_bug_speed(bug_traj)
                if self.pose_df is not None:
                    selected_frames = np.arange(strike_frame - n_pose_frames, strike_frame + n_pose_frames)
                    nose = self.pose_df.nose.loc[selected_frames, :].copy()
                    leap_frame_id, calc_strike_frame = self.calculate_leap_frame(nose.y, strike_frame)
                    pd = self.calculate_prediction_distance(leap_frame_id, i)
                data.append(self.pickle(i, locals()))

                if is_plot:
                    fig = plt.figure(figsize=(20, 15))
                    grid = fig.add_gridspec(ncols=4, nrows=3, width_ratios=[1, 1, 1, 1], height_ratios=[5, 6, 5])
                    fig.suptitle(f'Results for strike #{i + 1} ({"hit" if is_hit else "miss"})\n{self.loader}',
                                 fontsize=15)
                    self.plot_frames(grid[0, :], frame_group, frames, i, strike_frame)
                    if bug_traj is not None:
                        self.plot_bug_trajectory(fig.add_subplot(grid[1, 0]), bug_traj, i, leap_frame_id, bug_speed)
                        self.plot_projected_strike(fig.add_subplot(grid[1, 1]), bug_traj, i, pd, leap_frame_id, calc_strike_frame)
                    if nose is not None:
                        self.plot_xy_pose(fig.add_subplot(grid[1, 2]), strike_frame, nose)
                        self.plot_nose_vs_time(fig.add_subplot(grid[1, 3]), strike_frame, nose,
                                               leap_frame_id, calc_strike_frame)
                    self.plot_info(fig.add_subplot(grid[2, 0]))
                    self.plot_errors(fig.add_subplot(grid[2, 1:]))
                    fig.tight_layout()
                    fig.subplots_adjust(top=0.95, hspace=0.05)
                    fig.patch.set_linewidth(3)
                    fig.patch.set_edgecolor('black')

                    self.saved_image_folder.mkdir(exist_ok=True)
                    image_path = self.save_image_path(i)
                    fig.savefig(image_path)
                    images_paths.append(image_path.as_posix())
                    if is_close_plot:
                        plt.close(fig)

            except Exception as exc:
                self.log(f'Error in strike {i+1} {self.loader}; {exc}')

        return data, images_paths

    def plot_frames(self, grid, frame_group, frames, idx, strike_frame):
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
            if f_id == strike_frame:
                axes[i].set_title('strike frame')

    def plot_bug_trajectory(self, ax, bug_traj, i, leap_frame=None, bug_speed=0.0):
        try:
            cl = colorline(ax, bug_traj.x.to_numpy(), bug_traj.y.to_numpy(), alpha=1)
            plt.colorbar(cl, ax=ax, orientation='horizontal')
            ax.scatter(self.hits_df.x[i], self.hits_df.y[i], marker='X', color='r', label='strike', zorder=3)
            ax.add_patch(plt.Circle((self.hits_df.bug_x[i], self.hits_df.bug_y[i]),
                                    self.get_bug_radius(i), color='lemonchiffon', alpha=0.4))
            if leap_frame:
                leap_bug_traj = self.loader.bug_data_for_frame(leap_frame)
                if leap_bug_traj is not None:
                    ax.scatter(leap_bug_traj.x, leap_bug_traj.y, color='g', marker='D', s=50, label='bug at leap start',
                               zorder=2)
            ax.set_xlim([0, 2400])
            ax.set_ylim([-60, 1000])
            ax.set_title(f'Bug Trajectory\ncalculated speed: {bug_speed:.1f} cm/sec')
            ax.invert_yaxis()
            ax.legend()
        except Exception as exc:
            self.log(f'Error plotting bug trajectory; {exc}')

    def plot_projected_strike(self, ax, bug_traj, i, pd, leap_frame, default_leap=20, is_plot_strike_only=False):
        try:
            if leap_frame:
                leap_pos = self.loader.bug_data_for_frame(leap_frame)[['x', 'y']].to_numpy(dtype='float')
            else:
                leap_pos = bug_traj.loc[bug_traj.index[-default_leap], ['x', 'y']].to_numpy(dtype='float')
            hit = self.hits_df.loc[i, :]
            strike_pos, proj_leap_pos = self.project_strike_coordinates(hit, leap_pos)
            ax.scatter(*strike_pos, label='strike', color='r', marker='X')
            if is_plot_strike_only:
                return
            ax.scatter(0, 0, label='bug position at strike', color='m', marker='D')
            if leap_frame:
                ax.scatter(*proj_leap_pos, label='bug position at leap', color='g', marker='D')
            radius = self.get_bug_radius(i)
            ax.add_patch(plt.Circle((0, 0), radius, color='lemonchiffon', alpha=0.4))
            ax.legend()
            lim = max([*np.abs(strike_pos), *np.abs(proj_leap_pos), radius])
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

    @staticmethod
    def plot_xy_pose(ax, strike_frame_id, nose: pd.DataFrame):
        cl = colorline(ax, nose.x.to_numpy(), nose.y.to_numpy())
        plt.colorbar(cl, ax=ax, orientation='horizontal')
        ax.scatter(nose.x[strike_frame_id], nose.y[strike_frame_id], marker='X', color='r', label='strike', zorder=3)
        ax.legend()
        ax.set_title('XY of nose position')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    @staticmethod
    def plot_nose_vs_time(ax, strike_frame_id: int, nose: pd.DataFrame, leap_frame, calc_strike_frame):
        title = 'Y of nose vs. frame'
        ax.plot(nose.y, label='nose')
        ax.scatter(strike_frame_id, nose.y[strike_frame_id], marker='X', color='r', label='logged_strike')
        if calc_strike_frame is not None:
            ax.scatter(calc_strike_frame, nose.y[calc_strike_frame], marker='X', color='y', label='calc_strike')
            title += f'\nY(strike) = {nose.y[calc_strike_frame]:.2f}'
        if leap_frame is not None:
            ax.scatter(leap_frame, nose.y[leap_frame], marker='D', color='g', label='leap')
        if leap_frame is not None and calc_strike_frame is not None:
            title += f'\nLeap Frame Diff: {calc_strike_frame - leap_frame}'

        ax.legend()
        ax.set_title(title)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Y (nose)')

    def calculate_prediction_distance(self, leap_frame_id, i):
        leap_bug_traj = self.loader.bug_data_for_frame(leap_frame_id)
        if leap_bug_traj is None:
            return
        d = distance(leap_bug_traj.x, leap_bug_traj.y, self.hits_df.loc[i, 'x'], self.hits_df.loc[i, 'y'])
        return pixels2cm(d)

    @staticmethod
    def calculate_leap_frame(y: pd.DataFrame, strike_frame, th=0.9, grace=5, min_leap=10,
                             max_diff_strike_frame=8):
        peaks, _ = find_peaks(y.to_numpy(), height=870, distance=10)
        peaks_idx = y.index[peaks]
        peaks_idx = peaks_idx[np.abs(peaks_idx - strike_frame) < max_diff_strike_frame]
        try:
            if len(peaks_idx) > 0:
                strike_frame_idx = y[peaks_idx].idxmax()
            else:
                yrange = np.arange(strike_frame-max_diff_strike_frame,strike_frame+max_diff_strike_frame)
                strike_frame_idx = y[yrange].idxmax()
            dy = y.diff()
            leap_frame_idx = y.index[0]
            grace_count = 0
            for r in np.arange(strike_frame_idx, dy.index[0], -1):
                if dy[r] < th:
                    if grace_count == 0:
                        leap_frame_idx = r
                    grace_count += 1
                    if grace_count < grace:
                        continue
                    break
                else:
                    grace_count = 0
            if y[strike_frame_idx] - y[leap_frame_idx] < min_leap:
                return None, strike_frame_idx
        except Exception as exc:
            print(exc)
            return None, None

        return leap_frame_idx, strike_frame_idx

    @staticmethod
    def project_strike_coordinates(hits_df: pd.DataFrame, leap_pos):
        bug_pos = np.array([hits_df.bug_x, hits_df.bug_y])
        strike_pos = np.array([hits_df.x, hits_df.y])
        u = bug_pos - leap_pos
        xr = np.array([1, 0])
        th = np.arctan2(np.linalg.det([u, xr]), np.dot(u, xr))
        r = np.array(((np.cos(th), -np.sin(th)),
                      (np.sin(th), np.cos(th))))
        r0 = r.dot(bug_pos)
        return r.dot(strike_pos) - r0, r.dot(leap_pos) - r0

    def load_cached_summary(self, is_plot):
        data = []
        images_paths = []
        for i in self.hits_df.reset_index(drop=True).index:
            try:
                if is_plot:
                    plt.imshow(cv2.imread(self.save_image_path(i).as_posix()))
                images_paths.append(self.save_image_path(i))
                with self.save_pickle_path(i).open('rb') as f:
                    data.append(pickle.load(f))
            except Exception as exc:
                self.log(f'Error loading cached summary for strike {i+1}; {exc}')
        return data, images_paths

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

    def pickle(self, i, lcls):
        vars2pickle = ['bug_traj', 'strike_frame', 'leap_frame_id', 'calc_strike_frame', 'is_hit', 'bug_type',
                       'bug_radius', 'pd', 'bug_speed']
        bug_type = self.hits_df.loc[i, 'bug_type']
        bug_radius = self.get_bug_radius(i)
        lcls.update(locals())
        pickle_path = self.save_pickle_path(i)
        pickle_dict = {k: lcls.get(k) for k in vars2pickle}
        if pickle_dict.get('bug_speed'):
            pickle_dict['speed_group'] = self.get_speed_group(pickle_dict['bug_speed'])
        with pickle_path.open('wb') as f:
            pickle.dump(pickle_dict, f)

        return pickle_dict

    def save_image_path(self, i):
        return self.saved_image_folder / f'strike{i + 1}.jpg'

    def save_pickle_path(self, i):
        return self.saved_image_folder / f'strike{i + 1}.pickle'

    def plot_info(self, ax, w=30):
        ax.axis('off')
        text = f'Info:\n\n'
        for k, v in self.loader.info.items():
            if k.lower().startswith('block'):
                continue
            if isinstance(v, str) and len(v) > w:
                v_new = ''
                for i in range(0,len(v),w):
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

    def get_bug_trajectory_before_strike(self, i):
        try:
            return self.loader.get_bug_trajectory_before_strike(i, n_records=120, max_dist=0.4)
        except Exception as exc:
            self.log(str(exc))

    def get_bug_radius(self, i):
        bugs = {
            'cockroach': 150,
            'worm': 225,
            'red_beetle': 165,
            'black_beetle': 160,
            'green_beetle': 155
        }
        bug_type = self.hits_df.loc[i, 'bug_type']
        return bugs.get(bug_type, 160) / 1.5

    def get_pose(self):
        """Load the pose csv using pose.Analyzer"""
        a = Analyzer(self.loader.video_path)
        try:
            return a.run_pose(load_only=True)
        except Exception:
            return

    @staticmethod
    def get_speed_group(v):
        if 2 <= v < 6:
            return 4
        elif 6 <= v < 10:
            return 8
        elif 10 <= v < 14:
            return 12

    @staticmethod
    def calculate_bug_speed(bug_traj: pd.DataFrame):
        if bug_traj is None:
            return
        d = bug_traj.diff()
        v = np.sqrt(d.x ** 2 + d.y ** 2) / d.time.dt.total_seconds()
        return pixels2cm(v.mean())  # speed in cm/sec

    def log(self, msg):
        print(msg)
        self.strike_errors.append(msg)

    @property
    def saved_image_folder(self):
        return self.loader.trial_path / 'strike_analysis'


STRIKE_FIELDS = ['bug_type']


class MultiStrikesAnalyzer:
    def __init__(self, loaders, groupby=None, main=None, **filters):
        if groupby:
            assert isinstance(groupby, dict), 'groupby must be dictionary'
            assert all(isinstance(v, list) for v in groupby.values() if v is not None), \
                'all groupby values must be list or None'
        if main:
            assert main in groupby, f'main {main} is not in groupby'
        self.loaders = self.filter(loaders, **filters)
        self.main = main
        self.groupby = groupby
        self.info_df = self.load_data()

    def load_data(self):
        l = []
        fields2drop = ['bug_traj', 'nose']
        for ld in self.loaders:
            data, _ = StrikesSummary(ld).strikes_summary(is_plot=False, use_cache=True)
            for d in data:
                [d.pop(f, None) for f in fields2drop]
                d.update({k: v for k, v in ld.info.items() if not k.startswith('block')})
                l.append(d)

        return pd.DataFrame(l)

    @staticmethod
    def filter(loaders, **filters):
        if not filters:
            return loaders
        lds = []
        for ld in loaders:
            if all(ld.info.get(filter) == value for filter, value in filters.items()):
                lds.append(ld)
        print(f'Left with {len(lds)} loaders after applying the filters')
        return lds

    @staticmethod
    def create_subplots(n_groups):
        cols = min([4, n_groups or 1])
        rows = int(np.ceil((n_groups or 1) / cols))
        fig = plt.figure(figsize=(20, rows * 5))
        axes = fig.subplots(rows, cols)
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]
        return fig, axes

    @staticmethod
    def group_plot(plot_func, glds, ax, xlim, ylim, is_invert_y):
        for ld in glds:
            plot_func(ld, ax)
        ax.set_xlim(list(xlim))
        ax.set_ylim(list(ylim))
        if is_invert_y:
            ax.invert_yaxis()

    def subplot(self, plot_func, xlim=(0, 2300), ylim=(0, 900), is_invert_y=True):
        if not self.groupby:
            fig, axes = self.create_subplots(1)
            self.group_plot(plot_func, self.loaders, axes[0], xlim, ylim, is_invert_y)
            return

        groupby = list(self.groupby.keys())
        main_group = self.main or groupby[0]
        if len(groupby) > 1:
            groupby.remove(main_group)
        main_values = self.groupby[main_group] if self.groupby.get(main_group) else self.info_df[main_group].unique()
        for main_group_value in main_values:
            groups = self.info_df[self.info_df[main_group] == main_group_value].groupby(groupby).groups
            groups = self.check_groups(groups, groupby)
            fig, axes = self.create_subplots(len(groups))
            fig.suptitle(f'{main_group} = {main_group_value}', fontsize=15)
            for ia, (group_values, group_idx) in enumerate(groups.items()):
                if len(groupby) == 1:
                    group_values = [group_values]
                glds = [ld for j, ld in enumerate(self.loaders) if j in group_idx]
                self.group_plot(plot_func, glds, axes[ia], xlim, ylim, is_invert_y)
                if main_group != groupby[0]:
                    axes[ia].set_title(', '.join([f'{g}={v}' for g, v in zip(groupby, list(group_values))]))

    def check_groups(self, groups, groupby):
        new_groups = {}
        for group_values, group_idx in groups.items():
            is_group_ok = True
            for key, value in zip(groupby, group_values if len(groupby) > 1 else [group_values]):
                if self.groupby.get(key) and value not in self.groupby[key]:
                    is_group_ok = False
            if is_group_ok:
                new_groups[group_values] = group_idx

        return new_groups

    def plot_projected_strikes(self, xlim=(-200, 200), ylim=(-200, 200)):

        def _plot_projected_strikes(ld, ax):
            s = StrikesSummary(ld)
            for i in range(len(s.hits_df)):
                if 'bug_type' in self.groupby and not ax.patches:
                    ax.add_patch(plt.Circle((0, 0), s.get_bug_radius(i), color='lemonchiffon', alpha=0.4))
                pickle_path = s.save_pickle_path(i)
                if pickle_path.exists():
                    with pickle_path.open('rb') as f:
                        data = pickle.load(f)
                    if data.get('bug_traj') is None:
                        continue
                    s.plot_projected_strike(ax, data['bug_traj'], i, data.get('pd'), leap_frame=data.get('leap_frame'),
                                            is_plot_strike_only=True)
                    ax.plot([0, 0], ylim, 'k')
                    ax.plot(xlim, [0, 0], 'k')

        self.subplot(_plot_projected_strikes, xlim=xlim, ylim=ylim, is_invert_y=False)

    def plot_pd(self, xlim=(0, 15), ylim=(-2.5, 5)):

        def _plot_pd(ld, ax):
            s = StrikesSummary(ld)
            for i in range(len(s.hits_df)):
                pickle_path = s.save_pickle_path(i)
                if pickle_path.exists():
                    with pickle_path.open('rb') as f:
                        data = pickle.load(f)
                    if not data.get('pd') or not data.get('bug_speed'):
                        continue
                    ax.scatter(data['bug_speed'], data['pd'], color='b')
                    ax.set_xlabel('bug speed [cm/sec]')
                    ax.set_ylabel('PD [cm]')
        self.subplot(_plot_pd, xlim=xlim, ylim=ylim, is_invert_y=False)

    def plot_accuracy_vs_speed(self, xlim=(0, 15), ylim=(-2.5, 5)):

        def _plot_pd(ld, ax):
            s = StrikesSummary(ld)
            for i in range(len(s.hits_df)):
                pickle_path = s.save_pickle_path(i)
                if pickle_path.exists():
                    with pickle_path.open('rb') as f:
                        data = pickle.load(f)
                    if not data.get('pd') or not data.get('bug_speed'):
                        continue
                    ax.scatter(data['bug_speed'], data['pd'], color='b')
                    ax.set_xlabel('bug speed [cm/sec]')
                    ax.set_ylabel('PD [cm]')
        self.subplot(_plot_pd, xlim=xlim, ylim=ylim, is_invert_y=False)


def pixels2cm(x):
    return x * 0.01833304668870419


def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def transform_circle_center(x, y, x_center, y_center):
    x1 = x - x_center
    y1 = -y - y_center

    return x1, y1


def polar_transform(x, y, x_center, y_center):
    theta = lambda x1, y1: np.arctan2(x1, y1)
    rho = lambda x1, y1: np.sqrt(x1 ** 2 + y1 ** 2)
    x1, y1 = transform_circle_center(x, y, x_center, y_center)
    return theta(x1, y1), rho(x1, y1)
    # return np.array([theta(x, y) - theta(cx, cy), rho(x, y) - rho(cx, cy)])


def project(cx, cy, x, y) -> np.ndarray:
    th = np.arctan2(cx, cy)
    th = th - np.pi / 2
    r = np.array(((np.cos(th), -np.sin(th)),
                  (np.sin(th), np.cos(th))))
    u = np.array([x, y]) - np.array([cx, cy])
    xr = r.dot(np.array([1, 0]))
    yr = r.dot(np.array([0, 1]))
    projection = lambda x1, y1: y1 * np.dot(y1, x1) / np.dot(y1, y1)
    v = np.array([np.dot(projection(u, xr), xr), np.dot(projection(u, yr), yr)])
    if np.abs(v[0]) > 1000 or np.abs(v[1]) > 1000:
        return np.array([np.nan, np.nan])
    return v

def colorline(ax, x, y, z=None, cmap=plt.get_cmap('jet'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)
    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments
