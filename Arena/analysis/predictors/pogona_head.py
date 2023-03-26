from dataclasses import dataclass
import bbox
import cv2
import torch
import logging
import numpy as np
from analysis.predictors.base import Predictor
from analysis.predictors.yolov5.models.common import DetectMultiBackend
from analysis.predictors.yolov5.utils.torch_utils import select_device
from analysis.predictors.yolov5.utils.augmentations import letterbox
from analysis.predictors.yolov5.utils.general import check_img_size, non_max_suppression

THRESHOLD = 0.5
GRACE_PERIOD = 2  # seconds
MAX_GRACE = 5  # num frames
MIN_DISTANCE = 5  # cm
MAX_DISTANCE = 15  # pixels


class PogonaHead(Predictor):
    def __init__(self, cam_name):
        super(PogonaHead, self).__init__()
        self.cam_name = cam_name
        self.bodyparts = ['head']
        self.detector = YOLOv5Detector()
        self.is_initialized = False

    def init(self, img):
        self.detector.load()
        self.is_initialized = True

    def predict(self, img, return_centroid=True, is_draw_pred=False):
        det, img = self.detector.detect_image(img)
        if det is None:
            return None, None

        if return_centroid:
            return self.to_centroid(det)
        else:
            if is_draw_pred:
                img = self.draw_predictions(det, img)
            return det, img

    @staticmethod
    def draw_predictions(det, img):
        if det is None:
            return img
        xA, yA, xB, yB, confidence = det
        img = cv2.rectangle(img, (int(xA), int(yA)), (int(xB), int(yB)), (0, 255, 0), 2)
        return img

    @staticmethod
    def to_centroid(det):
        xA, yA, xB, yB, confidence = det
        return (xA + xB) / 2, (yA + yB) / 2


@dataclass
class YOLOv5Detector:
    weights_path: str = "analysis/predictors/yolov5/runs/train/exp/weights/best.pt"
    data_path: str = 'analysis/predictors/yolov5/data/coco128.yaml'
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
