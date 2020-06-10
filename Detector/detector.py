import torch
from Detector.models import Darknet
from utils.utils import load_classes, non_max_suppression

"""
- All detectors implement the function detect_image(), that return a (number of detections) X 5 Numpy array.
The (row - single detection) array format is left_x-left_y-width-height-confidence.
"""


class Detector_v3:
    def __init__(self,
                 model_def="Detector/Yolo3/config/yolov3-custom.cfg",
                 weights_path="Detector/Yolo3/weights/yolov3-pogonahead.pth",
                 class_path="Detector/Yolo3/classes.names",
                 img_size=416):
        
        self.model_def = model_def
        self.weights_path = weights_path
        self.class_path = class_path
        self.img_size = img_size

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
     
    
    def xyxy_to_xywh(self,xyxy, output_shape):
        """
        xyxy - an array of xyxy detections in input_size x input_size coordinates.
        output_shape - shape of output array (height, width)
        """

        
        input_size = self.img_size
        
        pad_x = max(output_shape[0] - output_shape[1], 0) * (input_size / max(output_shape))
        pad_y = max(output_shape[1] - output_shape[0], 0) * (input_size / max(output_shape))
        unpad_h = input_size - pad_y
        unpad_w = input_size - pad_x

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
        return torch.stack([x1, y1, box_w, box_h,xyxy[:,4]], dim=1)
    
    def detect_image(self, img,orig_width,orig_height, conf_thres=0.8, nms_thres=0.5):
        """
        Return yolo detection array for the supplied image.
        img - The image as a pytorch tensor. Expecting img_size x img_size dimensions.
        conf_thres - confidence threshold for detection
        nms_thres - threshold for non-max suppression
        """
        #image_tensor = torch.from_numpy(img)
        image_tensor = img.unsqueeze(0)
        if torch.cuda.is_available():
            input_img = image_tensor.type(torch.Tensor).cuda()
        else:
            input_img = image_tensor.type(torch.Tensor)
            
        # run inference on the model and get detections
        with torch.no_grad():
            detections = self.model(input_img)
            detections = non_max_suppression(detections,
                                             conf_thres, nms_thres)
        detections = detections[0]
        
        if detections is not None:
            return self.xyxy_to_xywh(detections,(orig_height,orig_width)).numpy()
        
        return None
    


    
    
