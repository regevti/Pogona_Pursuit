import deeplabcut
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import re
from pose import Analyzer, BODY_PARTS
from scipy.signal import find_peaks
from loader import Loader
from scipy import optimize

NUM_FRAMES_BACK = 60 * 5


class StrikesAnalyzer:
    def __init__(self, loader: Loader = None, experiment_name=None, trial_id=None, camera=None):
        self.loader = loader or Loader(experiment_name, trial_id, camera)
        self.pose_df = Analyzer(self.loader.video_path).run_pose()
        self.xfs = []

    def get_strikes_pose(self, n_frames_back=None, n_frames_forward=10):
        """Run pose estimation on hits frames"""
        n_frames_back = n_frames_back or NUM_FRAMES_BACK

        def first_frame(frame_id):
            return frame_id - n_frames_back if frame_id >= n_frames_back else 0
        
        def last_frame(frame_id):
            return frame_id + n_frames_forward if frame_id < len(self.pose_df) - n_frames_forward else self.pose_df.index[-1]
        
        frames_groups = [list(range(first_frame(f), last_frame(f))) for f in self.loader.get_hits_frames()]
        return self.get_pose_estimation(frames_groups)

    def get_pose_estimation(self, frames: list):
        flat_frames = sorted([item for sublist in frames for item in sublist])
        for frame_group in frames:
            self.xfs.append(self.pose_df.loc[frame_group, :])
        return self.xfs

    # def play_strike(self, xf: pd.DataFrame, n=20):
    #     frames2plot = xf.index[-n:]
    #     cols = 3
    #     rows = int(np.ceil(len(frames2plot) / cols))
    #     fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    #     axes = axes.flatten()
    #     for i, frame_id in enumerate(frames2plot):
    #         frame = self.pose_analyzer.saved_frames.get(frame_id)
    #         if frame is None:
    #             continue
    #         axes[i].imshow(frame, aspect="auto")
    #         for part in BODY_PARTS:
    #             axes[i].scatter(xf.loc[frame_id, part].x, xf.loc[frame_id, part].y)
    #         axes[i].set_title(frame_id)
    #     fig.tight_layout()
    #     plt.show()

    def ballistic_analysis(self):
        res = []
        hits_df = self.loader.hits_df
        for i, xf in enumerate(self.xfs):
            start_frame = self.get_strike_start_frame(xf)
            if start_frame is not None:
                strike_start_time = self.loader.frames_ts[start_frame]
                traj_time = self.loader.traj_df.time.dt.tz_convert('utc').dt.tz_localize(None)
                closest_hit_index = (traj_time - pd.to_datetime(strike_start_time)).abs().argsort()[0]
                bug_start_pos = self.loader.traj_df.loc[closest_hit_index, :]
                PD = distance(bug_start_pos.x, hits_df.loc[i, 'x'], bug_start_pos.y, hits_df.loc[i, 'y'])
                MD = distance(hits_df.loc[i, 'bug_x'], hits_df.loc[i, 'x'],
                              hits_df.loc[i, 'bug_y'], hits_df.loc[i, 'y'])
                res.append(PD / MD)
        return res

    def circle(self) -> pd.DataFrame:
        theta = lambda x, y: np.arctan2(y, x)
        projection = lambda x, y: y * np.dot(y, x) / np.dot(y, y)

        xc, yc, _ = fit_circle(self.loader.traj_df.x, - self.loader.traj_df.y)
        vs = []
        for i, m in enumerate(self.loader.hits_df):
            m = m.copy().reset_index(drop=True)
            m[['bug_x', 'x']] = m[['bug_x', 'x']] - xc
            m[['bug_y', 'y']] = -m[['bug_y', 'y']] - yc

            ths = theta(m.bug_x, m.bug_y)
            ths = ths - np.pi / 2
            for i, th in enumerate(ths):
                r = np.array(((np.cos(th), -np.sin(th)),
                              (np.sin(th), np.cos(th))))
                bug_u = np.array([m.bug_x[i], m.bug_y[i]])
                u = np.array([m.x[i], m.y[i]]) - bug_u
                xr = r.dot(np.array([1, 0]))
                yr = r.dot(np.array([0, 1]))
                v = np.array([np.dot(projection(u, xr), xr), np.dot(projection(u, yr), yr)])
                if np.abs(v[0]) > 1000 or np.abs(v[1]) > 1000:
                    continue
                vs.append(v)

        vs = pd.DataFrame(vs, columns=['x', 'y'])
        return vs

    @staticmethod
    def get_strike_start_frame(xf: pd.DataFrame, th=1.5, grace=3, min_leap=20) -> (int, None):
        y = xf['nose'].y
        peaks, _ = find_peaks(y.to_numpy(), height=840, distance=10)
        hit_idx = y.index[peaks][-1] if len(peaks) > 0 else y.index[-1]
        dy = y.diff()
        grace_count = 0
        first_idx = y.index[0]
        for r in np.arange(hit_idx, dy.index[0], -1):
            if dy[r] < th:
                grace_count += 1
                if grace_count < grace:
                    continue
                first_idx = r
                break
            else:
                grace_count = 0
        if y[hit_idx] - y[first_idx] < min_leap:
            return
        return first_idx


def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


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