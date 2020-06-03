import torch
from Detector.models import Darknet
from utils.utils import load_classes, non_max_suppression

default_model_def = "config/yolov3-custom.cfg"
default_weights_path = "weights/yolov3-pogonahead.pth"
default_class_path = "data/custom/classes.names"
default_img_size = 416

class Detector:
    def __init__(self,
                 model_def=default_model_def,
                 weights_path=default_weights_path,
                 class_path=default_class_path,
                 img_size=default_img_size):
        
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
 
    def detect_image(self, img, conf_thres=0.8, nms_thres=0.5):
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
        return detections[0]
