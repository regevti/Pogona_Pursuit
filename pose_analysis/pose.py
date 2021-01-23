from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import yaml
import subprocess
import shutil
from functools import lru_cache
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from dlclive import DLCLive, Processor
from matplotlib.colors import TABLEAU_COLORS, CSS4_COLORS
from loader import Loader
from pose_utils import colorline, calc_total_trajectory

DLC_PATH = '/data/pose_estimation/deeplabcut/projects/pogona_pursuit_realtime'
DLC_CONFIG_FILE = DLC_PATH + '/config.yaml'
ITERATION = 3
EXPORTED_MODEL_PATH = DLC_PATH + f'/exported-models/DLC_pogona_pursuit_resnet_50_iteration-{ITERATION}_shuffle-1'
PROBABILITY_THRESH = 0.85
BODY_PARTS = ['nose', 'left_ear', 'right_ear']
COLORS = list(TABLEAU_COLORS.values()) + list(CSS4_COLORS.values())


class PoseAnalyzer:
    def __init__(self, video_path):
        self.video_path = Path(video_path)
        self.loader = Loader(video_path=video_path, is_validate=False)
        self.dlc_live = DLCLive(EXPORTED_MODEL_PATH, processor=Processor())
        self.is_dlc_live_initiated = False
        self.saved_frames = {}
        self.video_out = None
        self.validate_video()
        self.dlc_config = self.load_dlc_config()

    def run_pose(self, selected_frames=None, is_save_frames=False, load_only=False) -> pd.DataFrame:
        """
        Run Pose Estimation
        :param selected_frames: List of frames IDs needed to be analyzed (ignore the rest)
        :param is_save_frames: True for saving frames in self.saved_frames
        :return: Dataframe with frames as index and body parts as columns
        """
        if selected_frames:
            selected_frames.sort()
        if not selected_frames:
            if self.output_video_path.exists():
                self.compress_video()
            if self.pose_df is not None:
                return self.pose_df
            elif load_only:
                raise Exception(f'cannot find video for {self.loader.experiment_name}, trial{self.loader.trial_id}')

        cap = cv2.VideoCapture(self.video_path.as_posix())
        res = []
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'start pose estimation for {self.loader.experiment_name} trial{self.loader.trial_id}')
        for frame_id in tqdm(range(num_frames)):
            ret, frame = cap.read()
            if selected_frames and frame_id not in selected_frames:
                continue
            if ret:
                if not self.is_dlc_live_initiated:
                    self.dlc_live.init_inference(frame)
                    self.is_dlc_live_initiated = True
                # Initialize video writer (only if no specific frames provided)
                if not selected_frames and self.video_out is None:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    print(f'Saving analyzed video to: {self.output_video_path}')
                    self.video_out = cv2.VideoWriter(self.output_video_path.as_posix(), fourcc, fps, (w, h))

                if is_save_frames:
                    self.saved_frames[frame_id] = frame
                    
                pred = self.dlc_live.get_pose(frame)
                pred_df = self.create_pred_df(pred, frame_id)
                res.append(pred_df)
                self.write_frame(frame, frame_id, pred_df)
            if not ret or (selected_frames and frame_id > selected_frames[-1]):
                break

        self.video_out.release()
        self.video_out = None
        self.compress_video()
        if res:
            df = pd.concat(res)
            df.to_csv(self.output_video_path.parent / (self.output_video_path.stem + '.csv'))
            return df

    def position_map(self, part='nose', is_plot=True):
        range = [[180, 1100], [650, 980]]
        df = self.run_pose(load_only=True)[part]
        df.dropna(inplace=True)
        hist = np.histogram2d(df.x, df.y, bins=(20, 20), range=range)
        if is_plot:
            plt.figure(figsize=(10, 10))
            plt.imshow(hist[0].T, extent=[x for sublist in range for x in sublist])
            # plt.hist2d(df.x, df.y, range=[[0, 1000], [0, 950]], bins=50) # , cmap='BuPu'
            plt.title(f"{self.loader.experiment_name}, trial{self.loader.trial_id}\nscreen here")
            plt.colorbar()
            plt.gca().invert_yaxis()
            info_st = '\n'.join([f'{k}: {v}' for k, v in self.loader.info.items()])
            plt.text(0, 0, str(info_st), wrap=True, ha='left', fontsize=14, color='w')
        return hist[0].T

    @property
    @lru_cache()
    def pose_df(self):
        if self.output_video_path.with_suffix('.mp4').exists() and self.output_csv_path.exists():
            return pd.read_csv(self.output_csv_path, index_col=0, header=[0, 1])

    def arena_trajectories(self, is_plot=True, ax=None, cmap=None, is_only_first=False, yrange=None,
                           min_total_traj=None, mode='bug_phases'):
        assert mode in ['bug_phases', 'first30']
        if self.pose_df is None:
            return
        if mode == 'bug_phases':
            starts, ends = self.loader.bug_phases()
        elif mode == 'first30':
            starts, ends = self.loader.first30_traj()
        if starts is None:
            print('No bug phases were found')
            return
        arena_trajs = []
        for start_t, end_t in zip(starts.time, ends.time):
            start_frame = self.loader.get_frame_at_time(start_t)
            end_frame = self.loader.get_frame_at_time(end_t)
            if start_frame and end_frame:
                try:
                    traj = self.pose_df.nose.loc[np.arange(start_frame, end_frame), :].copy()
                    if yrange is not None and len(traj.y[(yrange[0] <= traj.y) & (traj.y <= yrange[1])]) == 0:
                        # drop trajectories which are out of yrange
                        continue
                    if min_total_traj is not None and calc_total_trajectory(traj) < min_total_traj:
                        continue

                except Exception as exc:
                    print(exc)
                    continue
                arena_trajs.append(traj)
                if is_only_first:
                    break

        if is_plot:
            if ax is None:
                _, ax = plt.subplots(figsize=(15, 15))
            for i, traj in enumerate(arena_trajs):
                if not cmap:
                    cmap = 'Blues'
                    if i == 0:
                        cmap = 'Oranges'
                    elif i == len(arena_trajs) - 1:
                        cmap = 'Greys'
                cl = colorline(ax, traj.x.to_numpy(), traj.y.to_numpy(), alpha=1, cmap=plt.get_cmap(cmap))
                # print(len(ax.get_figure().axes))
                # if len(ax.collections) == 1:
                #     plt.colorbar(cl, ax=ax, orientation='vertical')
            ax.set_xlim([0, 2400])
            ax.set_ylim([0, 1000])

        return arena_trajs

    def write_frame(self, frame: np.ndarray, frame_id: int, pred_df: pd.DataFrame):
        try:
            frame = self.put_text(f'frame: {frame_id}', frame, 50, 50)
            bug_df = self.loader.bug_data_for_frame(frame_id)
            bug_position = f'({bug_df.x:.0f}, {bug_df.y:.0f})' if bug_df is not None else '-'
            frame = self.put_text(f'bug position: {bug_position}', frame, 50, 90)
            frame = self.plot_predictions(frame, frame_id, pred_df)
            
            self.video_out.write(frame)
        except Exception as exc:
            print(f'Error writing frame {frame_id}; {exc}')

    @staticmethod
    def put_text(text, frame, x, y, font_scale=1, color=(255, 255, 0), thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):
        """
        :param text: The text to put on frame
        :param frame: The frame numpy array
        :param x: x
        :param y: y
        :param font_scale:
        :param color: default: yellow (255,255,0)
        :param thickness: in px, default 2px
        :param font: font
        :return: frame with text
        """
        return cv2.putText(frame, str(text), (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    @staticmethod
    def plot_predictions(frame, frame_id, df):
        """scatter the body parts prediction dots"""
        for i, part in enumerate(df.columns.get_level_values(0).unique()):
            if df[part].isnull().values.any():
                continue
                
            cX = df[part]['x'][frame_id]
            cY = df[part]['y'][frame_id]
            color = tuple(int(COLORS[i][j:j+2], 16) for j in (1, 3, 5))
            cv2.circle(frame, (cX, cY), 5, color, -1)
        return frame

    def create_pred_df(self, pred, frame_id: int) -> pd.DataFrame:
        zf = pd.DataFrame(pred, index=self.dlc_config['bodyparts'], columns=['x', 'y', 'prob']) #.loc[BODY_PARTS, :]
        zf.loc[zf['prob'] < PROBABILITY_THRESH, ['x', 'y']] = np.nan
        s = pd.DataFrame(pd.concat([zf['x'], zf['y']]), columns=[frame_id]).T
        s.columns = pd.MultiIndex.from_product([['x', 'y'], zf.index]).swaplevel(0, 1)
        s.sort_index(axis=1, level=0, inplace=True)
        return s

    def compress_video(self):
        """Run H265 compression on pose video and remove the input pose avi file"""
        try:
            print(f'start H265 compression for {self.output_video_path}')
            vid_tmp = self.output_video_path.absolute().as_posix()
            subprocess.run(['ffmpeg', '-i', vid_tmp, '-c:v', 'libx265',
                            '-preset', 'fast', '-crf', '28', '-tag:v', 'hvc1',
                            '-c:a', 'eac3', '-b:a', '224k', vid_tmp.replace('.avi', '.mp4')])
            subprocess.run(['rm', '-f', vid_tmp])
        except Exception as exc:
            print(f'Error compressing video: {exc}')

    def validate_video(self):
        assert self.video_path.exists(), f'video {self.video_path.name} does not exist'
        assert self.video_path.suffix in ['.avi', '.mp4'], f'suffix {self.video_path.suffix} not supported'

    @staticmethod
    def load_dlc_config():
        return yaml.load(open(DLC_CONFIG_FILE), Loader=yaml.FullLoader)

    def save_cache(self, df):
        pass

    @property
    def output_dir(self):
        output_dir = self.video_path.parent / self.model_name
        if not output_dir.exists():
            output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    @property
    def output_csv_path(self):
        return self.output_dir / f'{self.loader.camera}.csv'

    @property
    def output_video_path(self):
        return self.output_dir / f'{self.loader.camera}.avi'

    @property
    def model_name(self):
        return Path(DLC_PATH).name + f'_iteration{ITERATION}'
