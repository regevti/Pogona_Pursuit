import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import cv2
from matplotlib import cm
from matplotlib.colors import TABLEAU_COLORS, CSS4_COLORS
from image_handlers.base_predictor import Predictor
from dlclive import DLCLive, Processor

COLORS = list(TABLEAU_COLORS.values()) + list(CSS4_COLORS.values())
DLC_FOLDER = '/media/sil2/Data/regev/pose_estimation/deeplabcut/projects/dlc_pogona_mini'
# EXPORTED_MODEL = DLC_FOLDER + '/exported-models/DLC_dlc_pogona_mini_resnet_50_iteration-2_shuffle-0/'
EXPORTED_MODEL = DLC_FOLDER + '/exported-models/DLC_dlc_pogona_mini_mobilenet_v2_1.0_iteration-2_shuffle-1/'
THRESHOLD = 0.8


class DeepLabCut(Predictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dlc_config = {}
        self.load_dlc_config()
        self.bodyparts = self.dlc_config['bodyparts']
        self.kp_colors = {k: COLORS[i] for i, k in enumerate(self.bodyparts)}
        self.processor = Processor()
        self.detector = DLCLive(EXPORTED_MODEL, processor=self.processor, resize=0.5)
        self.is_initialized = False

    def __str__(self):
        return f'deeplabcut-{self.cam_name}'

    def loop(self):
        self.logger.info('DeepLabCut was loaded')
        super().loop()

    def predict_frame(self, img):
        if not self.is_initialized:
            self.detector.init_inference(img)
            self.is_initialized = True

        pred = self.detector.get_pose(img)
        pdf = self.create_pred_df(pred)
        engagement_score = self.calc_engagement(pdf)
        img = self.annotate_image(img, pdf, engagement_score)
        # convert the predictions df to a dict with bodyparts as keys
        # and {'x': <>, 'y': <>, 'prob': <>} as values
        x = {k: d for k, d in zip(self.bodyparts, pdf.to_dict('records'))}
        return x, img

    def load_dlc_config(self):
        config_path = Path(DLC_FOLDER) / 'config.yaml'
        self.dlc_config = yaml.load(config_path.open(), Loader=yaml.FullLoader)

    def create_pred_df(self, pred) -> pd.DataFrame:
        zf = pd.DataFrame(pred, index=self.dlc_config['bodyparts'], columns=['x', 'y', 'prob']) #.loc[BODY_PARTS, :]
        # zf.loc[zf['prob'] < THRESHOLD, ['x', 'y']] = np.nan
        # s = pd.DataFrame(pd.concat([zf['x'], zf['y']]), columns=[frame_id]).T
        # s.columns = pd.MultiIndex.from_product([['x', 'y'], zf.index]).swaplevel(0, 1)
        # s.sort_index(axis=1, level=0, inplace=True)
        return zf

    def log_prediction(self, det, timestamp):
        if det is None:
            return
            # self.predictions.append((timestamp, None))
        else:
            # self.predictions.append((timestamp, det))
            return

    def calc_engagement(self, pdf):
        if not all([pdf.loc[obj, 'prob'] >= THRESHOLD for obj in ['nose', 'right_ear', 'left_ear']]):
            return

        y = pdf.loc['nose', 'y']
        theta = self.calc_head_angle(pdf)
        return self.angle_score(theta) * y

    @staticmethod
    def calc_head_angle(pdf):
        x_nose = pdf.loc['nose', 'x']
        y_nose = pdf.loc['nose', 'y']
        x_ears = (pdf.loc['right_ear', 'x'] + pdf.loc['left_ear', 'x']) / 2
        y_ears = (pdf.loc['right_ear', 'y'] + pdf.loc['left_ear', 'y']) / 2
        return np.arccos((x_nose - x_ears) / np.sqrt((x_nose-x_ears)**2 + (y_nose-y_ears)**2))

    @staticmethod
    def angle_score(x):
        mu = np.pi / 2
        s = (mu - np.pi / 4) / 2  # 2 SDs
        return np.exp(-(((x - np.pi / 2) ** 2) / (2 * s ** 2)))

    def annotate_image(self, frame, pdf, engagement_score):
        """scatter the body parts prediction dots"""
        for part, row in pdf.iterrows():
            if row.prob < THRESHOLD:
                continue

            color = tuple(int(self.kp_colors[part][j:j + 2], 16) for j in (1, 3, 5))
            frame = cv2.circle(frame, (int(row.x), int(row.y)), 5, color, -1)

        if engagement_score:
            font, color = cv2.FONT_HERSHEY_SIMPLEX, (255, 255, 0)
            text = f'Engagement: {engagement_score:.1f}'
            frame = cv2.putText(frame, text, (20, 10), font, 1, color, 2, cv2.LINE_AA)

        return frame
