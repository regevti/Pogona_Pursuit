import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.spatial import distance
import yaml
import cv2
import math
from sympy import symbols, solve, Eq
from dlclive import DLCLive, Processor
from dlclive.processor import KalmanFilterPredictor
from matplotlib.colors import TABLEAU_COLORS, CSS4_COLORS

from image_handlers.base_predictor import Predictor
from utils import run_in_thread

COLORS = list(TABLEAU_COLORS.values()) + list(CSS4_COLORS.values())
DLC_FOLDER = '/media/sil2/Data/regev/pose_estimation/deeplabcut/projects/dlc_pogona_mini'
EXPORTED_MODEL = DLC_FOLDER + '/exported-models/DLC_dlc_pogona_mini_resnet_50_iteration-2_shuffle-0/'
# EXPORTED_MODEL = DLC_FOLDER + '/exported-models/DLC_dlc_pogona_mini_mobilenet_v2_1.0_iteration-2_shuffle-1/'
# EXPORTED_MODEL = DLC_FOLDER + '/exported-models/DLC_dlc_pogona_mini_efficientnet-b2_iteration-2_shuffle-3'
# EXPORTED_MODEL = DLC_FOLDER + '/exported-models/DLC_dlc_pogona_mini_resnet_101_iteration-2_shuffle-8'
THRESHOLDS = {'nose': 0.55, 'right_ear': 0.55, 'left_ear': 0.55}
RELEVANT_BODYPARTS = ['nose'] #, 'right_ear', 'left_ear']
MIN_DISTANCE = 0.1  # cm
MIN_COMMIT_DISTANCE = 0.5  # cm
MAX_JUMP = 4  # cm. Maximum distance difference for 2 consecutive values.
PREDICTION_GRACE = 10  # number of preceded frames without prediction to be kept with the previous prediction


class DeepLabCut(Predictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dlc_config = {}
        self.load_dlc_config()
        self.bodyparts = self.dlc_config['bodyparts']
        self.kp_colors = {k: COLORS[i] for i, k in enumerate(self.bodyparts)}
        # self.processor = KalmanFilterPredictor()
        self.processor = Processor()
        self.detector = DLCLive(EXPORTED_MODEL, processor=self.processor)
        self.is_initialized = False
        self.head_angle = None
        self.cam_coords = {k: None for k in RELEVANT_BODYPARTS}
        self.positions = {k: None for k in RELEVANT_BODYPARTS}
        self.grace_counters = {k: 0 for k in RELEVANT_BODYPARTS}
        self.last_committed_pos = None

    def __str__(self):
        return f'deeplabcut-{self.cam_name}'

    def loop(self):
        self.logger.info('DeepLabCut was loaded')
        super().loop()

    def before_predict(self, img):
        if self.cam_config.get('is_color'):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.pred_image_size:
            img = cv2.resize(img, self.pred_image_size[:2][::-1])
            return img.reshape((*img.shape, 1))
        return img

    def predict_frame(self, img, timestamp):
        if not self.is_initialized:
            self.detector.init_inference(img)
            self.is_initialized = True
        pos = {}
        try:
            pred = self.detector.get_pose(img, frame_time=timestamp)
            pdf = self.create_pred_df(pred)
            displaced_parts = self.check_displacement(pdf)
            if displaced_parts:
                self.calc_engagement()
                pos = self.positions.copy()
                pos['displaced_parts'] = displaced_parts

            img = self.annotate_image(img)
        except Exception as exc:
            self.logger.exception(exc)
        return pos, img

    def create_pred_df(self, pred) -> pd.DataFrame:
        zf = pd.DataFrame(pred, index=self.dlc_config['bodyparts'], columns=['cam_x', 'cam_y', 'prob'])
        zf = zf.loc[RELEVANT_BODYPARTS, :]
        for bodypart in RELEVANT_BODYPARTS:
            if zf.loc[bodypart, 'prob'] < THRESHOLDS[bodypart]:
                zf.drop(bodypart, inplace=True)
        if zf.empty:
            return zf
        if self.caliber.is_on:
            zf[['x', 'y']] = zf[['cam_x', 'cam_y']].apply(lambda pos: self.caliber.get_location(*pos), axis=1).tolist()
        else:
            zf[['x', 'y']] = zf[['cam_x', 'cam_y']].copy()
        return zf

    def check_displacement(self, pdf):
        """validate if any of the bodyparts moved"""
        displaced_parts = []
        for bodypart in self.positions:
            prev_pos = self.positions.get(bodypart)
            if bodypart in pdf.index:
                row = pdf.loc[bodypart]
                current_pos = (float(row.x), float(row.y))
                if not prev_pos or self.is_displaced(current_pos, prev_pos):
                    self.positions[bodypart] = current_pos
                    self.cam_coords[bodypart] = (round(row.cam_x), round(row.cam_y))
                    displaced_parts.append(bodypart)
                self.grace_counters[bodypart] = 0
            else:  # bodypart wasn't detected
                if self.grace_counters[bodypart] > PREDICTION_GRACE:
                    self.positions[bodypart] = None
                    self.cam_coords[bodypart] = None
                    self.head_angle = None
                self.grace_counters[bodypart] += 1

        if 'nose' in displaced_parts and (('right_ear' in displaced_parts and not self.positions['left_ear']) or
                                          ('left_ear' in displaced_parts and not self.positions['right_ear'])):
            bodypart = 'left_ear' if not self.positions['left_ear'] else 'right_ear'
            prev_pos = self.positions.get(bodypart)
            current_pos = self.predict_missing_ear(pdf)
            if current_pos and (not prev_pos or self.is_displaced(current_pos, prev_pos)):
                displaced_parts.append(bodypart)

        return displaced_parts

    def log_prediction(self, pos, timestamp):
        if not pos:
            return
        displaced_parts = pos.pop('displaced_parts')
        pos.update({'head_angle': self.head_angle})
        self.predictions.append((timestamp, pos))
        if 'nose' in displaced_parts and \
                (not self.last_committed_pos or
                 distance.euclidean(self.positions['nose'], self.last_committed_pos) > MIN_COMMIT_DISTANCE):
            self.commit_pose(timestamp)
            self.last_committed_pos = self.positions['nose']

    @run_in_thread
    def commit_pose(self, timestamp):
        x, y = self.positions['nose']
        start_time = datetime.utcfromtimestamp(timestamp)
        self.orm.commit_pose_estimation(cam_name=self.cam_name, start_time=start_time, x=x, y=y,
                                        angle=self.head_angle, engagement=None, video_id=self.current_db_video_id)

    @staticmethod
    def is_displaced(current_pos, prev_pos):
        """Check new position and omit detections that are too close to previous detection or too far away"""
        dist = distance.euclidean(current_pos, prev_pos)
        return MIN_DISTANCE < dist < MAX_JUMP

    def calc_engagement(self):
        if not all([self.positions.get(x) for x in ['nose', 'right_ear', 'left_ear']]):
            return

        y = self.positions['nose'][1]
        self.head_angle = self.calc_head_angle()
        return self.angle_score(self.head_angle) * y

    def calc_head_angle(self):
        x_nose, y_nose = self.positions['nose']
        x_ears = (self.positions['right_ear'][0] + self.positions['left_ear'][0]) / 2
        y_ears = (self.positions['right_ear'][1] + self.positions['left_ear'][1]) / 2
        dy = y_ears - y_nose
        dx = x_ears - x_nose
        if dx != 0.0:
            theta = np.arctan(abs(dy) / abs(dx))
        else:
            theta = np.pi / 2
        if dx > 0:  # looking south
            theta = np.pi - theta
        if dy < 0:  # looking opposite the screen
            theta = -1 * theta
        if theta < 0:
            theta = 2 * np.pi + theta
        return theta

    @staticmethod
    def angle_score(x):
        mu = np.pi / 2
        s = (mu - np.pi / 4) / 2  # 2 SDs
        return np.exp(-(((x - np.pi / 2) ** 2) / (2 * s ** 2)))

    def predict_missing_ear(self, pdf):
        if 'right_ear' not in pdf.index:
            missing_ear, found_ear = 'right_ear', 'left_ear'
        elif 'left_ear' not in pdf.index:
            missing_ear, found_ear = 'left_ear', 'right_ear'
        else:
            return
        x1, y1 = pdf.loc[found_ear, ['cam_x', 'cam_y']]
        mean_dist_ears = 29  # pixels
        xn, yn = pdf.loc['nose', ['cam_x', 'cam_y']]
        x, y = symbols('x y')
        eq1 = Eq((x - x1) ** 2 + (y - y1) ** 2, mean_dist_ears ** 2)
        eq2 = Eq((x - xn) ** 2 + (y - yn) ** 2, (x1 - xn) ** 2 + (y1 - yn) ** 2)
        res = solve((eq1, eq2), (x, y), simplify=False, rational=False)

        pos = None
        for p in res:
            p = [float(z) for z in p if z.is_real]
            if not p:
                continue
            x2 = p[0]
            if (found_ear == 'left_ear' and ((y1 > yn and x2 > x1) or (y1 < yn and x2 < x1))) or \
                    (found_ear == 'right_ear' and ((y1 > yn and x2 < x1) or (y1 < yn and x2 > x1))):
                pos = p
                break

        if not pos:
            return
        current_pos = self.caliber.get_location(*pos)
        self.positions[missing_ear] = current_pos
        self.cam_coords[missing_ear] = (round(pos[0]), round(pos[1]))
        return current_pos

    def annotate_image(self, frame):
        """scatter the body parts prediction dots"""
        for part, pos in self.cam_coords.items():
            if not pos:
                continue
            # color = tuple(int(self.kp_colors[part][j:j + 2], 16) for j in (1, 3, 5))
            color = (0, 0, 0)
            frame = cv2.circle(frame, pos, 5, color, 3)

        text = ''
        font, color = cv2.FONT_HERSHEY_SIMPLEX, (255, 255, 0)
        if self.head_angle is not None:
            text += f'Angle={round(math.degrees(self.head_angle))}, '
        if self.positions.get('nose'):
            nose_pos = [round(z) for z in self.positions['nose']]
            text += f'loc={nose_pos}'
            frame = cv2.putText(frame, text, (20, 30), font, 1, color, 2, cv2.LINE_AA)

        return frame

    def load_dlc_config(self):
        config_path = Path(DLC_FOLDER) / 'config.yaml'
        self.dlc_config = yaml.load(config_path.open(), Loader=yaml.FullLoader)
