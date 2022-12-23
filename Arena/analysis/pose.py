import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
from dlclive import DLCLive, Processor
from matplotlib.colors import TABLEAU_COLORS, CSS4_COLORS
from tqdm.auto import tqdm
from pathlib import Path
import os
if Path('.').resolve().name != 'Arena':
    os.chdir('..')
from calibration import PoseEstimator
from loggers import get_logger
from db_models import ORM, Experiment, Block, Video, VideoPrediction, Strike

COLORS = list(TABLEAU_COLORS.values()) + list(CSS4_COLORS.values())
DLC_FOLDER = '/media/sil2/Data/regev/pose_estimation/deeplabcut/projects/dlc_pogona_mini'
EXPORTED_MODEL = DLC_FOLDER + '/exported-models/DLC_dlc_pogona_mini_resnet_50_iteration-2_shuffle-0/'
THRESHOLD = 0.5
RELEVANT_BODYPARTS = ['nose', 'right_ear', 'left_ear', 'forelegL1', 'forelegL2', 'hindlegL1', 'hindlegL2', 'forelegR1',
                      'forelegR2', 'hindlegR1', 'hindlegR2']


class DLCPose:
    def __init__(self, cam_name):
        self.logger = get_logger('DLC-Pose')
        self.cam_name = cam_name
        self.dlc_config = {}
        self.load_dlc_config()
        self.bodyparts = self.dlc_config['bodyparts']
        self.kp_colors = {k: COLORS[i] for i, k in enumerate(self.bodyparts)}
        self.processor = Processor()
        self.detector = DLCLive(EXPORTED_MODEL, processor=self.processor)
        self.is_initialized = False
        self.caliber = None

    def init(self, img):
        if self.is_initialized:
            return
        self.detector.init_inference(img)
        self.caliber = PoseEstimator(self.cam_name, resize_dim=img.shape[:2][::-1])
        self.caliber.init(img)
        self.is_initialized = True

    def predict(self, img, timestamp, frame_id=0):
        self.init(img)
        pdf = None
        try:
            pred = self.detector.get_pose(img, frame_time=timestamp)
            pdf = self.create_pred_df(pred, frame_id)
        except Exception as exc:
            self.logger.exception(exc)
        return pdf

    def predict_video(self, video_path, frame_timestamps, is_plot=False):
        if self.get_cache_path(video_path).exists():
            return
        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pdf = []
        for i in tqdm(range(n_frames), desc=Path(video_path).stem):
            ret, frame = cap.read()
            if not ret:
                break
            pdf_ = self.predict(frame, frame_timestamps[str(i)], i)
            if is_plot:
                frame = PredPlotter.plot_predictions(frame, i, pdf_)
                plt.imshow(frame)
                plt.show()
            pdf.append(pdf_)

        if pdf:
            pdf = pd.concat(pdf)
            self.save_cache(video_path, pdf)

        return pdf

    def create_pred_df(self, pred, frame_id) -> pd.DataFrame:
        cols = ['cam_x', 'cam_y', 'prob']
        zf = pd.DataFrame(pred, index=self.dlc_config['bodyparts'], columns=cols)
        zf = zf.loc[RELEVANT_BODYPARTS, :]
        if zf.empty:
            return zf

        zf, is_converted = self.convert_to_real_world(zf)
        if is_converted:
            cols = ['x', 'y'] + cols
        s = pd.DataFrame(pd.concat([zf[c] for c in cols]), columns=[frame_id]).T
        s.columns = pd.MultiIndex.from_product([cols, zf.index]).swaplevel(0, 1)
        s.sort_index(axis=1, level=0, inplace=True)
        return s

    def convert_to_real_world(self, zf):
        is_converted = False
        if self.caliber.is_on:
            zf[['x', 'y']] = zf[['cam_x', 'cam_y']].apply(lambda pos: self.caliber.get_location(*pos), axis=1).tolist()
            is_converted = True
        return zf, is_converted

    def add_real_world_values_to_existing_pose(self, video_path, pose_df):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        self.caliber = PoseEstimator(self.cam_name, resize_dim=frame.shape[:2])
        self.caliber.init(frame)
        if self.caliber.state != 2:
            self.caliber.find_aruco_markers(frame)
            self.caliber.init(frame)
        pose_df, is_converted = self.convert_to_real_world(pose_df)
        return pose_df

    def load_dlc_config(self):
        config_path = Path(DLC_FOLDER) / 'config.yaml'
        self.dlc_config = yaml.load(config_path.open(), Loader=yaml.FullLoader)

    def save_cache(self, video_path, pdf: pd.DataFrame):
        cache_path = self.get_cache_path(video_path)
        pdf.to_parquet(cache_path)

    @staticmethod
    def get_cache_path(video_path) -> Path:
        preds_dir = Path(video_path).parent / 'predictions'
        preds_dir.mkdir(exist_ok=True)
        return preds_dir / Path(video_path).with_suffix('.parquet').name


class PredPlotter:
    @classmethod
    def plot_predictions(cls, frame, frame_id, df, parts2plot=None):
        """scatter the body parts prediction dots"""
        x_legend, y_legend = 30, 30
        for i, part in enumerate(df.columns.get_level_values(0).unique()):
            if parts2plot and part not in parts2plot:
                continue
            elif part in ['time', 'tongue'] or df[part].isnull().values.any() or df[part]['prob'].loc[frame_id] < THRESHOLD:
                continue

            cX = round(df[part]['cam_x'][frame_id])
            cY = round(df[part]['cam_y'][frame_id])
            color = tuple(int(COLORS[i][j:j + 2], 16) for j in (1, 3, 5))
            cv2.circle(frame, (cX, cY), 5, color, -1)

            cv2.circle(frame, (x_legend, y_legend), 5, color, -1)
            cls.put_text(part, frame, x_legend + 20, y_legend)
            y_legend += 30
        return frame

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


if __name__ == '__main__':
    orm = ORM()
    desired_cam_name, desired_animal_id = 'front', 'PV80'
    dp = DLCPose(desired_cam_name)
    with orm.session() as s:
        for exp in s.query(Experiment).filter_by(animal_id=desired_animal_id).all():
            for blk in exp.blocks:
                for vid in blk.videos:
                    if vid.cam_name != desired_cam_name:
                        continue
                    try:
                        dp.predict_video(vid.path, vid.frames)
                    except Exception as exc:
                        print(f'ERROR; {vid.path}; {exc}')
