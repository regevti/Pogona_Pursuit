import numpy as np
import cv2 as cv
from ctypes import c_int, pointer

import Prediction.Yolo4.darknet as darknet4
import torch


class Detector:
    """
    Abstract class that detects bounding boxes in image data.
    Subclasses should override the detect_image(self, img) method.
    Abstract class with a detect_image() method that takes an image numpy array and
    returns a (number of detections) X 5 numpy array.
    Each row represents one detection as [left_x, top_y, right_x, bottom_y, confidence] of
    the bounding box, where (0,0) is the top left corner of the image.
    """

    def detect_image(self, img):
        """
        Detect objects in the supplied image and return an Nx5 numpy array of bounding boxes + confidence.
        Each row represents one detection as [left_x, top_y, right_x, bottom_y, confidence] of
        the bounding box, where (0,0) is the top left corner of the image.

        img - The image as a numpy array (cv2 frame etc.)

        Return the detection array for the supplied image.
        """
        pass


class Detector_v4:
    """
    Detector using version 4 of the YOLO algorithm, based on the paper "YOLOv4: Optimal Speed and Accuracy of Object Detection"
    Code from: https://github.com/AlexeyAB/darknet, including a python wrapper for the C modules.
    Training was done with "non-examples", i.e frames from the arena with no detections and unrelated images.

    The resized image used for the last detection is stored in self.curr_img
    """

    def __init__(self,
                 cfg_path="Prediction/Yolo4/yolo4_2306.cfg",
                 weights_path="Prediction/Yolo4/yolo4_gs_best_2306.weights",
                 meta_path="Prediction/Yolo4/obj.data",
                 conf_thres=0.9,
                 nms_thres=0.6):
        """
        Instantiate detector.
        cfg_path - Path to yolo network configuration file
        weights_path - Path to trained network weights
        meta_path - Path to yolo metadata file (pretty useless for inference but necessary)
        conf_thres - Confidence threshold for bounding box detections
        nms_thres - Non-max suppression threshold. Suppresses multiple detections for the same object.
        """
        self.net = darknet4.load_net_custom(cfg_path.encode("ascii"),
                                            weights_path.encode("ascii"),
                                            0, 1)
        self.meta = darknet4.load_meta(meta_path.encode("ascii"))
        self.model_width = darknet4.lib.network_width(self.net)
        self.model_height = darknet4.lib.network_height(self.net)
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.curr_img = None
        print("Detector initiated successfully")

    def set_conf_and_nms(self, new_conf_thres=0.9, new_nms_thres=0.6):
        """
        Set new confidence threshold and nms threshold values.
        """
        self.conf_thres = new_conf_thres
        self.nms_thres = new_nms_thres

    def detect_image(self, img):
        """
        Receive an image as numpy array. Resize image to model size using open-cv.
        Run the image through the network and collect detections.
        Return a numpy array of detections. Each row is x1, y1, x2, y1, confidence 
        (top-left and bottom-right corners).
        """
        input_height, input_width, _ = img.shape

        image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (self.model_width, self.model_height), interpolation=cv.INTER_LINEAR)
        self.curr_img = image
        image, arr = darknet4.array_to_image(image)
        num = c_int(0)
        pnum = pointer(num)
        darknet4.predict_image(self.net, image)

        dets = darknet4.get_network_boxes(self.net, input_width, input_height,
                                          self.conf_thres, self.conf_thres, None, 0, pnum, 0)

        num = pnum[0]
        if self.nms_thres:
            darknet4.do_nms_sort(dets, num, self.meta.classes, self.nms_thres)

        res = np.zeros((num, 5))
        for i in range(num):
            b = dets[i].bbox
            res[i] = [b.x - b.w / 2, b.y - b.h / 2, b.x + b.w / 2, b.y + b.h / 2, dets[i].prob[0]]
        nonzero = res[:, 4] > 0
        res = res[nonzero]

        darknet4.free_detections(dets, num)

        if res.shape[0] == 0:
            return None
        else:
            return res


def xywh_to_centroid(xywh):
    """
    Return the centroids of a bbox array in xywh values (1 or 2 dimensional).
    xywh - bbox array in x, y, width, height.
    """
    if len(xywh.shape) == 1:
        x, y, w, h = xywh[:4]
        return np.array([x + w // 2, y + h // 2])

    x1 = xywh[:, 0]
    y1 = xywh[:, 1]
    box_w = xywh[:, 2]
    box_h = xywh[:, 3]

    return np.stack([x1 + (box_w // 2), y1 + (box_h // 2)], axis=1)


def xywh_to_xyxy(xywh):
    """
    Convert a numpy array of bbox coordinates from xywh to xyxy

    xywh - bbox array in x, y, width, height

    Return the bbox in xyxy coordinates - [x1, y1, x2, y2] (top-left, bottom-right corners)
    """
    if len(xywh.shape) == 1:
        x, y, w, h = xywh[:4]
        return np.array([x, y, x + w, y + h])

    x1 = xywh[:, 0]
    y1 = xywh[:, 1]
    box_w = xywh[:, 2]
    box_h = xywh[:, 3]

    return np.stack([x1, y1, x1 + box_w, y1 + box_h], axis=1)


def xyxy_to_xywh(xyxy):
    """
    Convert a numpy array of bbox coordinates from xyxy to xywh

    :param xywh: bboxes in x, y, width, height format
    :return: bboxes in x,y, width, height format
    """
    if len(xyxy.shape) == 1:
        x1, y1, x2, y2 = xyxy[:4]
        return np.array([x1, y1, (x2 - x1), (y2 - y1)])

    x1 = xyxy[:, 0]
    y1 = xyxy[:, 1]
    x2 = xyxy[:, 2]
    y2 = xyxy[:, 3]

    return np.stack([x1, y1, (x2 - x1), (y2 - y1)], axis=1)


def xyxy_to_centroid(xyxy):
    """
    Convert a numpy array of bbox coordinates (xyxy) to an array of bbox centroids.
    Return an array of centroids, each row consisting of x, y centroid coordinates.

    xyxy - bbox array in x1, y1, x2, y2
    """
    if len(xyxy.shape) == 1:
        x1, y1, x2, y2 = xyxy[:4]
        return np.array([(x2 + x1) / 2, (y2 + y1) / 2])

    x1 = xyxy[:, 0]
    y1 = xyxy[:, 0]
    x2 = xyxy[:, 2]
    y2 = xyxy[:, 3]

    return np.stack([(x1 + x2) / 2, (y1 + y2) / 2], axis=1)


def centwh_to_xyxy(centwh):
    """
    Convert a numpy array of bbox coordinates in center x, y, width, height format to xyxy format

    :param centwh: bboxes in xywh format where x, y are the centroid coordinates
    :return: bboxes in xyxy format (top-left, bottom-right corners)
    """
    if type(centwh) == list or len(centwh.shape) == 1:
        cx, cy, w, h = centwh[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    cx = centwh[:, 0]
    cy = centwh[:, 1]
    w = centwh[:, 2]
    h = centwh[:, 3]

    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def nearest_detection(detections, prev_centroid):
    """
    Return the detection from the detections array whose centroid is closest to the previous centroid.
    When only one detection is supplied this detection is returned.

    detections - A numpy detections array.
    prev_centroid - The centroid of a previous detection (x, y) to compare to.

    The returned detection is a single row from the detections array.
    """
    if detections.shape[0] > 1:
        detected_centroids = xyxy_to_centroid(detections)
        deltas = prev_centroid - detected_centroids
        dists = np.linalg.norm(deltas, axis=1)
        arg_best = np.argmin(dists)
        return detections[arg_best]
    else:
        return detections[0]


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou
