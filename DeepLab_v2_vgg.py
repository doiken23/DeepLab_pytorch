import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import math
from collections import OrderedDict

def conv3x3_relu(inplanes, planes, rate=1):
    conv3x3_relu = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3, 
                                    stride=1, padding=rate, dilation=rate),
                                 nn.ReLU())
    return conv3x3_relu

class VGG16_feature(nn.Module):
    def __init__(self, pretrained=False):
        super(VGG16_feature, self).__init__()

        self.features = nn.Sequential(conv3x3_relu(3, 64),
                                      conv3x3_relu(64, 64),
                                      nn.MaxPool2d(2, stride=2),
                                      conv3x3_relu(64, 128),
                                      conv3x3_relu(128, 128),
                                      nn.MaxPool2d(2, stride=2),
                                      conv3x3_relu(128, 256),
                                      conv3x3_relu(256, 256),
                                      conv3x3_relu(256, 256),
                                      nn.MaxPool2d(2, stride=2),
                                      conv3x3_relu(256, 512),
                                      conv3x3_relu(512, 512),
                                      conv3x3_relu(512, 512),
                                      nn.MaxPool2d(3, stride=1, padding=1))
        self.features2 = nn.Sequential(conv3x3_relu(512, 512, rate=2),
                                       conv3x3_relu(512, 512, rate=2),
                                       conv3x3_relu(512, 512, rate=2),
                                       nn.MaxPool2d(3, stride=1, padding=1))

        """
        if pretrained:
            url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
            weight  = model_zoo.load_url(url)
            weight2 = OrderedDict()
            for key in list(weight.keys())[:20]:
                weight2[key] = weight[key]

            self.features.load_state_dict(weight2)
        """

    def forward(self, x):
        x = self.features(x)
        x = self.features2(x)

        return x


class Atrous_module(nn.Module):
    def __init__(self, inplanes, num_classes, rate):
        super(Atrous_module, self).__init__()
        planes = inplanes
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=3,
                                            stride=1, padding=rate, dilation=rate)
        self.fc1 = nn.Conv2d(planes, planes, kernel_size=1, stride=1)
        self.fc2 = nn.Conv2d(planes, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


class DeepLabv1_ASPP(nn.Module):
    def __init__(self, num_classes, small=True, pretrained=False):
        super(DeepLabv2_ASPP, self).__init__()
        self.vgg_classifier = VGG16_feature(pretrained)

        if small:
            rates = [2, 4, 8, 12]
        else:
            rates = [6, 12, 18, 24]
        self.aspp1 = Atrous_module(2048 , num_classes, rate=rates[0])
        self.aspp2 = Atrous_module(2048 , num_classes, rate=rates[1])
        self.aspp3 = Atrous_module(2048 , num_classes, rate=rates[2])
        self.aspp4 = Atrous_module(2048 , num_classes, rate=rates[3])
        
    def forward(self, x):
        x = self.resnet_classifier(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        x = x1 + x2 + x3 + x4
        x = F.upsample(x, scale_factor=8, mode='bilinear')

        return x 

class DeepLabv1_FOV(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(DeepLabv2_FOV, self).__init__()
        self.vgg_classifier = VGG16_feature(pretrained)

        self.atrous = Atrous_module(2048 , num_classes, rate=12)
        
    def forward(self, x):
        x = self.resnet_classifier(x)
        x = self.atrous(x)
        x = F.upsample(x, scale_factor=8, mode='bilinear')

        return x 
