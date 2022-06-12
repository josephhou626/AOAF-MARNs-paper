import torch
import torch.nn as nn
from model.sp import StripPooling

class MARNs(nn.Module) :


    def __init__(self):
        super(MARNs,self).__init__()
        self.cnn = nn.Sequential(
            #layer1
            nn.Conv2d(1,64,3,1,1),
            nn.ReLU(),
            StripPooling(64,(12,20),nn.BatchNorm2d,{"mode":"bilinear"}),
            #layer2
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(),
            #layer3
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(),
            #layer4
            StripPooling(64,(12,20),nn.BatchNorm2d,{"mode":"bilinear"}),
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(),
            #layer5
            StripPooling(64,(12,20),nn.BatchNorm2d,{"mode":"bilinear"}),
            nn.Conv2d(64,1,3,1,1),
            nn.ReLU(),
        )


    def forward(self,x):
        out = self.cnn(x)
        return out