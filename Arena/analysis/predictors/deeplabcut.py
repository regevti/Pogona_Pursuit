import os
import cv2
import warnings
import pandas as pd
from pathlib import Path
import numpy as np
import yaml
import argparse
from matplotlib.colors import TABLEAU_COLORS, CSS4_COLORS
if __name__ == '__main__':
    os.chdir('../..')

import config
from analysis.predictors.base import Predictor
from analysis.pose_utils import put_text

COLORS = list(TABLEAU_COLORS.values()) + list(CSS4_COLORS.values())


class DLCPose(Predictor):
    def __init__(self, cam_name):
        super().__init__()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            from dlclive import DLCLive, Processor
        self.cam_name = cam_name
        self.dlc_config = {}
        self.load_dlc_config()
        self.bodyparts = self.dlc_config['bodyparts'] + ['mid_ears']
        self.kp_colors = {k: COLORS[i] for i, k in enumerate(self.bodyparts)}
        self.processor = Processor()
        self.detector = DLCLive(self.model_path, processor=self.processor)
        self.is_initialized = False

    def init(self, img):
        if self.is_initialized:
            return
        self.detector.init_inference(img)
        self.is_initialized = True

    def predict(self, img, frame_id=0) -> pd.DataFrame:
        pdf = self._predict(img, frame_id)
        for bodypart in self.bodyparts:
            df_ = pdf.loc[frame_id, bodypart]
            if df_['prob'] < self.threshold:
                pdf.iloc[0][(bodypart, 'cam_x')] = None
                pdf.iloc[0][(bodypart, 'cam_y')] = None
        return pdf

    def _predict(self, img, frame_id=0):
        self.init(img)
        pdf = None
        try:
            pred = self.detector.get_pose(img)
            pdf = self.create_pred_df(pred, frame_id)
        except Exception as exc:
            print(exc)
        return pdf

    def create_pred_df(self, pred, frame_id) -> pd.DataFrame:
        cols = ['cam_x', 'cam_y', 'prob']
        zf = pd.DataFrame(pred, index=self.bodyparts[:-1], columns=cols)
        zf.loc['mid_ears', :] = zf.loc[['left_ear', 'right_ear'], :].mean()
        zf = zf.loc[self.bodyparts, :]
        if zf.empty:
            return zf

        s = pd.DataFrame(pd.concat([zf[c] for c in cols]), columns=[frame_id]).T
        s.columns = pd.MultiIndex.from_product([cols, zf.index]).swaplevel(0, 1)
        s.sort_index(axis=1, level=0, inplace=True)
        return s

    def plot_predictions(self, frame, frame_id, df, parts2plot=None):
        """scatter the body parts prediction dots"""
        x_legend, y_legend = 30, 30
        parts2plot = parts2plot or self.bodyparts
        if frame_id not in df.index:
            return
        for i, part in enumerate(df.columns.get_level_values(0).unique()):
            if not part or (parts2plot and part not in parts2plot):
                continue
            elif df[part].loc[frame_id].isnull().values.any() or df[part]['prob'].loc[frame_id] < self.threshold:
                continue

            cX = round(df[part]['cam_x'][frame_id])
            cY = round(df[part]['cam_y'][frame_id])
            color = tuple(int(COLORS[i][j:j + 2], 16) for j in (1, 3, 5))
            cv2.circle(frame, (cX, cY), 8, color, -1)

            cv2.circle(frame, (x_legend, y_legend), 5, color, -1)
            put_text(part, frame, x_legend + 20, y_legend)
            y_legend += 30
        return frame

    @staticmethod
    def plot_single_part(df, frame_id, frame):
        if np.isnan(df['x'][frame_id]):
            return frame
        cX = round(df['x'][frame_id])
        cY = round(df['y'][frame_id])
        color = tuple(int(COLORS[1][j:j + 2], 16) for j in (1, 3, 5))
        cv2.circle(frame, (cX, cY), 7, color, -1)
        return frame

    def load_dlc_config(self):
        config_path = Path(config.DLC_FOLDER) / 'config.yaml'
        self.dlc_config = yaml.load(config_path.open(), Loader=yaml.FullLoader)


# class DLCVideoCache:
#     def __init__(self, bodyparts):
#         self.bodyparts = bodyparts
#
#     def load_pose_df(self, vid: Video = None,
#                      video_path: str = None,
#                      frames_df: pd.DataFrame = None,
#                      keypoint: str = None):
#         assert not (vid is not None and video_path), 'must only provide one of the two: vid, video_path'
#         if vid is not None:
#             video_path = Path(vid.path).resolve()
#         elif video_path:
#             video_path = Path(video_path)
#             vid = self.get_video_db(video_path)
#         assert video_path.exists(), f'Video {video_path} does not exist'
#
#         pred_path = self.get_cache_path(video_path)
#         if not pred_path.exists():
#             raise Exception(f'could not find pose predictions file in {pred_path}')
#         pdf = pd.read_parquet(pred_path)
#         if frames_df is None:
#             frames_df = self.load_frames_times(vid)
#         if len(frames_df) > len(pdf):
#             frames_df = frames_df.iloc[:len(pdf)]
#         elif 1 <= (len(pdf) - len(frames_df)) <= 10:
#             pdf = pdf.iloc[:len(frames_df)]
#         if len(frames_df) != len(pdf):
#             raise Exception(f'different length to frames_df ({len(frames_df)}) and pose_df ({len(pdf)})')
#         frames_df.columns = pd.MultiIndex.from_tuples([('time', '')])
#         pdf = pd.concat([frames_df, pdf], axis=1)
#         if keypoint:
#             pdf = pd.concat([pdf.time, pdf[keypoint]], axis=1)
#         pdf, _ = self.convert_to_real_world(pdf)
#         return pdf
#
#
#
#     @staticmethod
#     def get_cache_path(video_path) -> Path:
#         preds_dir = Path(video_path).parent / 'predictions'
#         preds_dir.mkdir(exist_ok=True)
#         return preds_dir / Path(video_path).with_suffix('.parquet').name
#
#     def save_cache(self, video_path, pdf: pd.DataFrame):
#         cache_path = self.get_cache_path(video_path)
#         pdf.to_parquet(cache_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deeplabcut helper')
    parser.add_argument('command', choices=['export', 'train'], help='helper command')
    args = parser.parse_args()

    if args.command == 'export':
        name = input('>> please enter name for the new model: ')
        assert all(c not in name for c in '$!@=+/`~:;±§()*%&'), f'{name} - invalid name'

    elif args.command == 'train':
        print('train')