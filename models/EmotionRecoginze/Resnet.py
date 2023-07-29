import numpy as np
import random
import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets,models

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Res18(nn.Module):
    def __init__(self, pretrained=True,inplanes=128 , num_classes=8, drop_rate=0.0):
        super(Res18, self).__init__()
        self.drop_rate = drop_rate
        self.inplanes = inplanes
        resnet = models.resnet18(pretrained)
        # self.feature = nn.Sequential(*list(resnet.children())[:-2]) # before avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # after avgpool 512x1

        fc_in_dim = list(resnet.children())[-1].in_features  # original fc layer's in dimention 512

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.fc = nn.Linear(fc_in_dim, num_classes)  # new fc layer 512x7
        self.sigmod = nn.Sigmoid()

    def forward(self, x):

        x = self.features(x)

        if self.drop_rate > 0:
            x = nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)
        fc_weight = x
        x = self.fc(x)
        out = self.sigmod(x)
        return out,fc_weight


class Dense121(nn.Module):
    def __init__(self, pretrained=True,inplanes=128 , num_classes=8, drop_rate=0.0):
        super(Dense121, self).__init__()
        self.drop_rate = drop_rate
        self.inplanes = inplanes
        densenet = models.densenet121(pretrained=pretrained)
        # self.feature = nn.Sequential(*list(resnet.children())[:-2]) # before avgpool
        self.features = nn.Sequential(*list(densenet.children())[:-1])  # after avgpool 512x1

        fc_in_dim = list(densenet.children())[-1].in_features  # original fc layer's in dimention 512
        # print(list(densenet.children()))

        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        self.fc = nn.Linear(fc_in_dim, num_classes) # new fc layer 512x7
        self.sigmod = nn.Sigmoid()

    def forward(self, x):

        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        # if self.drop_rate > 0:
        #     x = nn.Dropout(self.drop_rate)(x)
        x = self.fc(x)
        out = self.sigmod(x)
        return out