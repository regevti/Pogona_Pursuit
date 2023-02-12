import numpy as np
import cv2
from dataclasses import dataclass
import bbox
import torch
import torch.nn as nn
from torchvision import transforms, models
from image_handlers.predictor_handlers import PredictHandler


class ResnetEmbedding(PredictHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detector = ResNetPretrained()
        self.detector.eval()
        self.last_embedding = None

    def __str__(self):
        return f'pogona-head-{self.cam_name}'

    def loop(self):
        self.logger.info('Resnet was loaded')
        super().loop()

    def predict_frame(self, img):
        with torch.no_grad():
            _, x = self.detector(img)
            x = x.cpu().numpy().astype('float').tolist()
        return x, img

    def log_prediction(self, det, timestamp):
        if det is None:
            # self.predictions.append((timestamp, None, None, None))
            return
        else:
            self.predictions.append((timestamp, det))


class ResNetPretrained(nn.Module):
    def __init__(self, rescale_size=(224, 224)):
        super().__init__()
        resnet = models.resnet101(pretrained=True)
        resnet.float()
        resnet.cuda()
        resnet.eval()
        module_list = list(resnet.children())
        self.conv5 = nn.Sequential(*module_list[:-2])
        self.pool5 = module_list[-2]
        self.transformer = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(rescale_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        x = self.transformer(x).unsqueeze(0).cuda()
        res5c = self.conv5(x)
        pool5 = self.pool5(res5c)
        pool5 = pool5.view(pool5.size(0), -1)
        return res5c, pool5

    def feature_extraction(self, frames: np.ndarray):
        """
        Extract features using resnet
        @param frames: [n_frames, 3, 224, 224]
        @return: embedded features [n_frames, 2040]
        """
        features = []
        for frame in frames:
            _, x = self(frame)
            features.append(x.detach().cpu().numpy())
        return np.vstack(features)