import logging
import numpy as np
import cv2
from dataclasses import dataclass
import bbox
import torch
from image_handlers.base_predictor import Predictor

from image_handlers.yolov5.models.common import DetectMultiBackend
from image_handlers.yolov5.utils.torch_utils import select_device
from image_handlers.yolov5.utils.augmentations import letterbox
from image_handlers.yolov5.utils.general import check_img_size, non_max_suppression


class PogonaHeadDetector(Predictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detector = YOLOv5Detector(return_neareast_detection=False, logger=self.logger)
        self.detector.load()

    def __str__(self):
        return f'pogona-head-{self.cam_name}'

    def loop(self):
        self.logger.info(
            f"YOLOv5 detector loaded successfully ({self.detector.model_width}x{self.detector.model_height} "
            f"weights: {self.detector.weights_path})."
        )
        super().loop()

    def predict_frame(self, img, timestamp):
        det, img = self.detector.detect_image(img)
        return det, img

    def log_prediction(self, det, timestamp):
        if det is None:
            return
            # self.predictions.append((timestamp, None, None, None))
        else:
            xA, yA, xB, yB, confidence = det[0, :].flatten()
            x_center, y_center = (xA + xB) / 2, (yA + yB) / 2
            self.predictions.append((timestamp, float((xA + xB) / 2), float((yA + yB) / 2), float(confidence)))


@dataclass
class YOLOv5Detector:
    weights_path: str = "image_handlers/yolov5/runs/train/exp/weights/best.pt"
    data_path: str = 'image_handlers/yolov5/data/coco128.yaml'
    model_width: int = 640
    model_height: int = 480
    conf_thres: float = 0.5
    iou_thres: float = 0.45
    device: str = 'cuda:0'
    return_neareast_detection: bool = False
    logger: logging.Logger = None

    def __post_init__(self):
        self.device = select_device(self.device)

    def load(self):
        self.prev_bbox = None
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
        res = pred.cpu().numpy()[:, :5]

        if self.return_neareast_detection:
            if self.prev_bbox is None:
                self.prev_bbox = res[np.argmax(res[:, 4])]
            else:
                self.prev_bbox = bbox.nearest_bbox(res, bbox.xyxy_to_centroid(self.prev_bbox))
            return self.prev_bbox, image
        return res, self.draw_prediction_onto_image(image, res)

    def draw_prediction_onto_image(self, img, res):
        xA, yA, xB, yB, confidence = res[0, :].flatten()
        img = cv2.rectangle(img, (int(xA), int(yA)), (int(xB), int(yB)), (0, 255, 0), 2)
        # img = cv2.putText(img, str(confidence), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # img = img.transpose((2, 1, 0))
        return img
