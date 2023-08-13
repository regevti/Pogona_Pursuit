import numpy as np
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights


class ResNetPretrained(nn.Module):
    def __init__(self, is_grey=False):
        super().__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        if is_grey:
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.float()
        resnet.cuda()
        resnet.eval()
        module_list = list(resnet.children())
        self.conv5 = nn.Sequential(*module_list[:-2])
        self.pool5 = module_list[-2]

    def forward(self, x):
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
