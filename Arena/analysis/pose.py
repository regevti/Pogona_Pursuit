import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import yaml
from dlclive import DLCLive, Processor
from matplotlib.colors import TABLEAU_COLORS, CSS4_COLORS
from tqdm.auto import tqdm
from pathlib import Path
import os
if Path('.').resolve().name != 'Arena':
    os.chdir('..')
import config
from calibration import PoseEstimator
from loggers import get_logger
from sqlalchemy import cast, Date
from db_models import ORM, Experiment, Block, Video, VideoPrediction, Strike, PoseEstimation

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
        self.orm = ORM()

    def init(self, img):
        if self.is_initialized:
            return
        self.detector.init_inference(img)
        self.init_calibrator(img)
        self.is_initialized = True

    def init_calibrator(self, img, img_shape=None):
        img_shape = img_shape or img.shape[:2][::-1]
        self.caliber = PoseEstimator(self.cam_name, resize_dim=img_shape)
        self.caliber.init(img, img_shape=img_shape)

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

    def load_pose_df(self, vid: Video = None,
                     video_path: str = None,
                     frames_df: pd.DataFrame = None,
                     keypoint: str = None):
        assert not (vid is not None and video_path), 'must only provide one of the two: vid, video_path'
        if vid is not None:
            video_path = Path(vid.path).resolve()
        elif video_path:
            video_path = Path(video_path)
            vid = self.get_video_db(video_path)
        assert video_path.exists(), f'Video {video_path} does not exist'
        pred_path = self.get_cache_path(video_path)
        if not pred_path.exists():
            raise Exception(f'could not find pose predictions file in {pred_path}')
        pdf = pd.read_parquet(pred_path)
        if frames_df is None:
            frames_df = self.load_frames_times(vid)
        if len(frames_df) > len(pdf):
            frames_df = frames_df.iloc[:len(pdf)]
        elif 1 <= (len(pdf) - len(frames_df)) <= 10:
            pdf = pdf.iloc[:len(frames_df)]
        if len(frames_df) != len(pdf):
            raise Exception(f'different length to frames_df ({len(frames_df)}) and pose_df ({len(pdf)})')
        frames_df.columns = pd.MultiIndex.from_tuples([('time', '')])
        pdf = pd.concat([frames_df, pdf], axis=1)
        if keypoint:
            pdf = pd.concat([pdf.time, pdf[keypoint]], axis=1)
        pdf, _ = self.convert_to_real_world(pdf)
        return pdf

    def get_video_db(self, video_path: Path):
        with self.orm.session() as s:
            vid = s.query(Video).filter(Video.path.contains(video_path.stem)).first()
        return vid

    def load_frames_times(self, vid: Video) -> pd.DataFrame:
        frames_df = pd.DataFrame()
        if vid.frames is None:
            print(f'{str(self)} - frame times does not exist')
            return frames_df
        frames_ts = pd.DataFrame(vid.frames.items(), columns=['frame_id', 'time']).set_index('frame_id')
        frames_ts['time'] = pd.to_datetime(frames_ts.time, unit='s', utc=True).dt.tz_convert('Asia/Jerusalem').dt.tz_localize(None)
        frames_ts.index = frames_ts.index.astype(int)
        return frames_ts

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


class SpatialAnalyzer:
    def __init__(self, animal_id, day=None, cam_name=None):
        self.animal_id = animal_id
        self.day = day
        self.cam_name = cam_name

    def query_pose(self):
        orm = ORM()
        with orm.session() as s:
            q = s.query(PoseEstimation).filter_by(animal_id=self.animal_id)
            if self.cam_name:
                q = q.filter_by(cam_name=self.cam_name)
            q = q.filter(cast(PoseEstimation.start_time, Date) == self.day)
            res = q.all()
        return res

    def get_bug_exit_hole(self):
        orm = ORM()
        with orm.session() as s:
            q = s.query(Block).filter(cast(Block.start_time, Date) == self.day)
            res = q.all()
        return set([r.exit_hole for r in res])

    def plot_spatial(self, pose=None):
        if pose is None:
            res = self.query_pose()
            if res:
                print('No pose recordings found')
                return
            pose = pd.DataFrame([(r.x, r.y) for r in res], columns=['x', 'y'])

        fig, ax = plt.subplots(1, 1, figsize=(10, 20))
        ax.scatter(pose['x'], pose['y'])
        rect = patches.Rectangle((0, -2), 42, 80, linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(rect)
        screen = patches.Rectangle((2, -3), 38, 2, linewidth=1, edgecolor='k', facecolor='k')
        ax.add_patch(screen)
        ax.set_title(f'{self.animal_id}, {self.day}, {self.get_bug_exit_hole()}')
        plt.show()


def get_day_from_path(p):
    return p.stem.split('_')[1].split('T')[0]


def load_pose_from_videos(animal_id, cam_name, day=None):
    dlc_pose = DLCPose(cam_name=cam_name)
    dlc_pose.init_calibrator(None, (1088, 1456))
    exp_dir = Path(config.experiments_dir) / animal_id
    reg = f'{cam_name}_*.mp4' if not day else f'{cam_name}_{day}T*.mp4'
    for p in exp_dir.rglob(reg):
        if dlc_pose.get_cache_path(p).exists():
            df = dlc_pose.load_pose_df(video_path=p, keypoint='nose')
            day_ = day or get_day_from_path(p)
            sa = SpatialAnalyzer(animal_id, day_, cam_name)
            sa.plot_spatial(df)


if __name__ == '__main__':
    # img = cv2.imread('/data/Pogona_Pursuit/output/calibrations/front/20221205T094015_front.png')
    # plt.imshow(img)
    # plt.show()
    load_pose_from_videos('PV80', 'front', day='20221211')
    # SpatialAnalyzer('PV80', day='2022-12-15').plot_spatial()

    # orm = ORM()
    # desired_cam_name, desired_animal_id = 'front', 'PV80'
    # dp = DLCPose(desired_cam_name)
    # with orm.session() as s:
    #     for exp in s.query(Experiment).filter_by(animal_id=desired_animal_id).all():
    #         for blk in exp.blocks:
    #             for vid in blk.videos:
    #                 if vid.cam_name != desired_cam_name:
    #                     continue
    #                 try:
    #                     dp.predict_video(vid.path, vid.frames)
    #                 except Exception as exc:
    #                     print(f'ERROR; {vid.path}; {exc}')
