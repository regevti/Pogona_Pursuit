import time
import logging
import numpy as np
import cv2
from dataclasses import dataclass
import bbox
import pandas as pd
from scipy.spatial import distance
import torch
from image_handlers.base_predictor import Predictor
from utils import run_in_thread, KalmanFilter

from image_handlers.yolov5.models.common import DetectMultiBackend
from image_handlers.yolov5.utils.torch_utils import select_device
from image_handlers.yolov5.utils.augmentations import letterbox
from image_handlers.yolov5.utils.general import check_img_size, non_max_suppression

THRESHOLD = 0.5
GRACE_PERIOD = 2  # seconds
MAX_GRACE = 5  # num frames
VELOCITY_SAMPLING_DURATION = 2  # seconds
MIN_DISTANCE = 5  # cm
MAX_DISTANCE = 15  # pixels


class PogonaHeadDetector(Predictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detector = YOLOv5Detector(return_neareast_detection=False, logger=self.logger)
        self.detector.load()
        self.velocity = None
        self.last_det = None
        self.kalman = KalmanFilter()
        self.grace_count = 0

    def __str__(self):
        return f'pogona-head-{self.cam_name}'

    def loop(self):
        self.logger.info(
            f"YOLOv5 detector loaded successfully ({self.detector.model_width}x{self.detector.model_height} "
            f"weights: {self.detector.weights_path})."
        )
        super().loop()

    def predict_frame(self, img, timestamp):
        """Get detection of pogona head on frame; {det := [x1, y1, x2, y2, confidence]}"""
        det, img = self.detector.detect_image(img)
        return det, img

    def log_prediction(self, det, timestamp):
        if det is None:
            return

        xA, yA, xB, yB, confidence = det
        x, y = self.to_centroid(xA, yA, xB, yB)
        if self.caliber.is_on:
            x, y = self.caliber.get_location(x, y)
        x, y = self.kalman.get_filtered((x, y))
        self.predictions.append((timestamp, x, y, float(confidence)))

    def analyze_predictions(self, predictions: list, current_db_video_id, predictions_start_time):
        t0 = time.time()
        predictions = np.array(predictions)
        x, y = [round(z) for z in predictions[:, 1:3].mean(axis=0)]
        d = predictions[:, :3]
        if len(d) > 1:
            d = np.diff(d, axis=0)
        v = (np.sqrt(d[:, 1]**2 + d[:, 2]**2) / d[:, 0]).mean()
        self.velocity = v

        if self.last_commit and distance.euclidean(self.last_commit[1:], (x, y)) < MIN_DISTANCE:
            return

        self.orm.commit_pose_estimation(self.cam_name, predictions_start_time, x, y, None, None, current_db_video_id)
        self.last_commit = (t0, x, y)

        #
        # if det is None:
        #     if self.last_det is not None and time.time() - self.last_det[4] > GRACE_PERIOD:
        #         self.last_det = None
        #     return
        #
        # x, y, confidence = self.analyze_position(det, timestamp)
        #
        #
        # self.predictions.append((timestamp, x, y, float(confidence)))
        # if len(self.predictions) > 5:
        #     self.calc_velocity()
        # else:
        #     self.velocity = None

    # def analyze_position(self, det, timestamp):
    #     xA, yA, xB, yB, confidence = det
    #     x, y = self.to_centroid(xA, yA, xB, yB)
    #
    #     dist, x0, y0 = None, None, None
    #     #
    #     if self.last_det is not None:
    #         timestamp0 = self.last_det[4]
    #         if time.time() - timestamp0 <= GRACE_PERIOD:
    #             x0, y0 = self.to_centroid(*self.last_det[:4])
    #             dist = distance.euclidean((x0, y0), (x, y))
    #
    #     if dist is not None and (dist < MIN_DISTANCE or dist > MAX_DISTANCE):
    #         # in cases where the displacement is too small or too large (bad detection), use the previous detection
    #         x, y = x0, y0
    #     else:
    #         self.last_det = np.append(det, timestamp)
    #     return x, y, confidence

    def calc_velocity(self):
        df = pd.DataFrame(self.predictions, columns=['time', 'x', 'y', 'confidence'])
        df = df.query(f'time > {time.time() - VELOCITY_SAMPLING_DURATION}').copy()
        df.loc[:, 'dist'] = np.sqrt(df.x.diff() ** 2 + df.y.diff() ** 2)
        df.loc[:, 'velocity'] = df.dist / df.time.diff()
        self.velocity = df.velocity.mean()

    @run_in_thread
    def commit_to_db(self):
        pass

    def draw_pred_on_image(self, det, img, font=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 0, 0)):
        h, w = img.shape[:2]
        # for z in [3, 2, 1.5]:
        #     x, y = self.caliber.get_location(round(w / z), round(h / 2))
        #     img = cv2.putText(img, f'{(round(x), round(y))}', (round(w / z), round(h / 2)), font, 0.5, color, 2, cv2.LINE_AA)

        if det is None:
            # if self.last_det and time.time() - self.last_det[4] < GRACE_PERIOD:
            #     det = self.last_det
            # else:
            return img

        xA, yA, xB, yB, confidence = det
        img = cv2.rectangle(img, (int(xA), int(yA)), (int(xB), int(yB)), (0, 255, 0), 2)
        if self.predictions:
            _, x, y, _ = self.predictions[-1]
            img = cv2.putText(img, f'loc:{round(x)},{round(y)}', (20, h-30), font, 1, color, 2, cv2.LINE_AA)
        self.last_det = det
        # if self.velocity:
        #     units = 'cm/sec' if self.caliber.is_on else 'pixels/sec'
        #     img = cv2.putText(img, f'Velocity: {self.velocity:.1f}{units}', (20, 30), font, 1, color, 2, cv2.LINE_AA)
        return img

    @staticmethod
    def to_centroid(xA, yA, xB, yB):
        return (xA + xB) / 2, (yA + yB) / 2


@dataclass
class YOLOv5Detector:
    weights_path: str = "image_handlers/yolov5/runs/train/exp/weights/best.pt"
    data_path: str = 'image_handlers/yolov5/data/coco128.yaml'
    model_width: int = 640
    model_height: int = 480
    conf_thres: float = THRESHOLD
    iou_thres: float = 0.45
    device: str = 'cuda:0'
    return_neareast_detection: bool = False
    logger: logging.Logger = None

    def __post_init__(self):
        self.device = select_device(self.device)

    def load(self):
        # due to issue of pytorch model load in fork-multiprocessing, the model must be loaded in CPU
        # and only then be deployed to CUDA
        self.model = DetectMultiBackend(self.weights_path, device=select_device('cpu'), dnn=True, data=self.data_path)
        self.stride = self.model.stride
        self.imgsz = check_img_size((self.model_height, self.model_width), s=self.stride)  # check image size
        self.model.warmup(imgsz=(1, 3, *self.imgsz))  # warmup
        self.model.to(self.device)

    @torch.no_grad()
    def detect_image(self, image):
        """
        Bounding box inference on input image
        :param img: numpy array image
        :return: list of detections. Each row is x1, y1, x2, y2, confidence  (top-left and bottom-right corners).
        """
        img = letterbox(image, self.imgsz, stride=self.stride, auto=True)[0]
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        image = img.copy()
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        im = torch.from_numpy(img).to(self.device).float()
        im /= 255  # 0 - 255 to 0.0 - 1.0

        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        with torch.no_grad():
            pred = self.model(im, augment=False, visualize=False)
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=False, max_det=1000)

        pred = pred[0]
        if len(pred) == 0:
            return None, image
        return pred.cpu().numpy().flatten()[:5], image
