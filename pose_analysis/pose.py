from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import yaml
import subprocess
import shutil
from tqdm.auto import tqdm
from dlclive import DLCLive, Processor
from matplotlib.colors import TABLEAU_COLORS
from loader import Loader

DLC_PATH = '/data/pose_estimation/deeplabcut/projects/pogona_pursuit-regev-2020-07-19'
DLC_CONFIG_FILE = DLC_PATH + '/config.yaml'
EXPORTED_MODEL_PATH = DLC_PATH + '/exported-models/DLC_pogona_pursuit_resnet_50_iteration-1_shuffle-1'
PROBABILITY_THRESH = 0.85
BODY_PARTS = ['nose', 'left_ear', 'right_ear']
COLORS = list(TABLEAU_COLORS.values())


class Analyzer:
    def __init__(self, video_path):
        self.video_path = Path(video_path)
        self.loader = Loader(video_path=video_path)
        self.dlc_live = DLCLive(EXPORTED_MODEL_PATH, processor=Processor())
        self.is_dlc_live_initiated = False
        self.saved_frames = {}
        self.video_out = None
        self.validate_video()
        self.dlc_config = self.load_dlc_config()

    def run_pose(self, frames=None, is_save_frames=False) -> pd.DataFrame:
        """
        Run Pose Estimation
        :param frames: List of frames IDs needed to be analyzed (ignore the rest)
        :param is_save_frames: True for saving frames in self.saved_frames
        :return: Dataframe with frames as index and body parts as columns
        """
        if frames:
            frames.sort()
        if not frames:
            if self.output_video_path.exists():
                self.compress_video()
            if self.output_video_path.with_suffix('.mp4').exists() and self.output_csv_path.exists():
                print(f'video and csv already exist in {self.output_dir}')
                return pd.read_csv(self.output_csv_path, index_col=0, header=[0, 1])

        cap = cv2.VideoCapture(self.video_path.as_posix())
        res = []
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'start pose estimation for {self.loader.experiment_name} trial{self.loader.trial_id}')
        for frame_id in tqdm(range(num_frames)):
            ret, frame = cap.read()
            if frames and frame_id not in frames:
                continue
            if ret:
                if not self.is_dlc_live_initiated:
                    self.dlc_live.init_inference(frame)
                    self.is_dlc_live_initiated = True
                # Initialize video writer (only if no specific frames provided)
                if not frames and self.video_out is None:
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
            if not ret or (frames and frame_id > frames[-1]):
                break

        self.video_out.release()
        self.video_out = None
        if res:
            df = pd.concat(res)
            df.to_csv(self.output_video_path.parent / (self.output_video_path.stem + '.csv'))
            return df

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
    def put_text(text, frame, x, y, font_scale=1, color=(255,255,0), thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):
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
        zf = pd.DataFrame(pred, index=self.dlc_config['bodyparts'], columns=['x', 'y', 'prob']).loc[BODY_PARTS, :]
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
        return Path(EXPORTED_MODEL_PATH).name
