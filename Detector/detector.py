import torch
from torchvision import transforms
from PIL import Image
import Detector.Yolo4.darknet as darknet4
from Detector.models import Darknet
from utils.utils import load_classes, non_max_suppression
import cv2 as cv
from ctypes import c_int, pointer
import numpy as np

"""
All detectors implement the function detect_image(), that return a (number of detections) X 5 Numpy array.
The (row - single detection) array format is left_x-left_y-width-height-confidence.
"""


class Detector:
    def detect_image(self, img):
        """
        Return detection array for the supplied image.
        img - The image as a numpy array (cv2 frame etc.)
        """
        pass


class Detector_v3(Detector):
    """
    Yolo-V3 detector implemented in Pytorch. Training and model code taken from https://github.com/eriklindernoren/PyTorch-YOLOv3.
    based on original paper "YOLOv3: An Incremental Improvement".
    This model training was done without non-examples, i.e, images that do not contain any ground truth detection
    """
    
    def __init__(self,
                 model_def="Detector/Yolo3/config/yolov3-custom.cfg",
                 weights_path="Detector/Yolo3/weights/yolov3-pogonahead.pth",
                 class_path="Detector/Yolo3/classes.names",
                 img_size=416,
                 conf_thres=0.9,
                 nms_thres=0.6):

        self.model_def = model_def
        self.weights_path = weights_path
        self.class_path = class_path
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            print("WARNING: GPU is not available")

        # Initiate model
        self.model = Darknet(self.model_def, img_size=self.img_size)
        self.model.load_state_dict(torch.load(weights_path, map_location=device))
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()
        self.classes = load_classes(class_path)
     
    def set_input_size(self, width, height):
        self.input_width = width
        self.input_height = height

        # update transforms: scale and pad image
        ratio = min(self.img_size/width, self.img_size/height)
        imw = round(width * ratio)
        imh = round(height * ratio)
        # resize + pad can be done with open cv and numpy!!
        self.resize_transform = transforms.Compose([transforms.Resize((imh, imw)),
                                                    transforms.Pad((max(int((imh - imw)/2), 0),
                                                                    max(int((imw - imh)/2), 0), max(int((imh-imw)/2), 0),
                                                                    max(int((imw - imh)/2), 0)), (128, 128, 128)),
                                                    transforms.ToTensor()])

    def xyxy_to_xywh(self, xyxy, output_shape):
        """
        xyxy - an array of xyxy detections in input_size x input_size coordinates.
        output_shape - shape of output array (height, width)
        """
        pad_x = max(output_shape[0] - output_shape[1], 0) * (self.img_size / max(output_shape))
        pad_y = max(output_shape[1] - output_shape[0], 0) * (self.img_size / max(output_shape))
        unpad_h = self.img_size - pad_y
        unpad_w = self.img_size - pad_x

        x1 = xyxy[:, 0]
        y1 = xyxy[:, 1]
        x2 = xyxy[:, 2]
        y2 = xyxy[:, 3]

        box_h = ((y2 - y1) / unpad_h) * output_shape[0]
        box_w = ((x2 - x1) / unpad_w) * output_shape[1]
        y1 = ((y1 - pad_y // 2) / unpad_h) * output_shape[0]
        x1 = ((x1 - pad_x // 2) / unpad_w) * output_shape[1]

        # return detections as (num_detections)X5 tensor, with
        # format xywh-conf
        return torch.stack([x1, y1, box_w, box_h, xyxy[:, 4]], dim=1)

    def detect_image(self, img):
        """
        Return yolo detection array for the supplied image.
        img - The image as numpy array.
        conf_thres - confidence threshold for detection
        nms_thres - threshold for non-max suppression
        """
        
        # TODO - could be updated to be done without conversion to PIL image
        # might be faster
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        PIL_img = Image.fromarray(img_rgb)
        image_tensor = self.resize_transform(PIL_img).unsqueeze(0)

        if torch.cuda.is_available():
            input_img = image_tensor.type(torch.Tensor).cuda()
        else:
            input_img = image_tensor.type(torch.Tensor)

        # run inference on the model and get detections
        with torch.no_grad():
            detections = self.model(input_img)
            detections = non_max_suppression(detections,
                                             self.conf_thres,
                                             self.nms_thres)
        detections = detections[0]

        if detections is not None:
            return self.xyxy_to_xywh(detections, (self.input_height, self.input_width)).numpy()

        return None


class Detector_v4:
    """
    Version 4 of the YOLO algorithm, based on the paper "YOLOv4: Optimal Speed and Accuracy of Object Detection"
    Code from: https://github.com/AlexeyAB/darknet, including a python wrapper for the C modules
    Training was done with "non-examples", i.e frames from the arena with no detections and unrelated images
    """
    def __init__(self,
                 cfg_path="Detector/Yolo4/yolo-obj.cfg",
                 weights_path="Detector/Yolo4/yolo-obj_best.weights",
                 meta_path="Detector/Yolo4/obj.data",
                 conf_thres=0.9,
                 nms_thres=0.6):

        self.net = darknet4.load_net_custom(cfg_path.encode("ascii"),
                                            weights_path.encode("ascii"),
                                            0, 1)
        self.meta = darknet4.load_meta(meta_path.encode("ascii"))
        self.model_width = darknet4.lib.network_width(self.net)
        self.model_height = darknet4.lib.network_height(self.net)
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        print("Detector initiated successfully")
  
    def set_input_size(self, width, height):
        self.input_width = width
        self.input_height = height
    
    def set_conf_and_nms(self,new_conf_thres=0.9,new_nms_thres=0.6):
        self.conf_thres = new_conf_thres
        self.nms_thres = new_nms_thres
    
    def detect_image(self, img):
        """
        Receive an image as numpy array. Resize image to model size using open-cv.
        Run the image through the network and collect detections.
        Return a numpy array of detections. Each row is x, y, w, h (top-left corner).
        """
        image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (self.model_width, self.model_height), interpolation=cv.INTER_LINEAR)
        image, arr = darknet4.array_to_image(image)

        num = c_int(0)
        pnum = pointer(num)
        darknet4.predict_image(self.net, image)

        dets = darknet4.get_network_boxes(self.net, self.input_width, self.input_height,
                                          self.conf_thres, self.conf_thres, None, 0, pnum, 0)

        num = pnum[0]
        if self.nms_thres:
            darknet4.do_nms_sort(dets, num, self.meta.classes, self.nms_thres)

        res = np.zeros((num, 5))
        for i in range(num):
            b = dets[i].bbox
            res[i] = [b.x-b.w/2, b.y-b.h/2, b.w, b.h, dets[i].prob[0]]
        nonzero = res[:, 4] > 0
        res = res[nonzero]

        darknet4.free_detections(dets, num)

        if res.shape[0] == 0:
            return None
        else:
            return res
