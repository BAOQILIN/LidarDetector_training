import torch.nn as nn
import numpy as np


class Deblock(nn.Module):
    def __init__(self, input_channel, filter_num, upsample_stride):
        super().__init__()
        if upsample_stride >= 1:
            self.model = nn.Sequential(nn.ConvTranspose2d(input_channel,
                                                          filter_num,
                                                          upsample_stride,
                                                          stride=upsample_stride,
                                                          bias=False),
                                       nn.BatchNorm2d(filter_num, eps=1e-3, momentum=0.01),
                                       nn.ReLU())
        else:
            stride = np.round(1 / upsample_stride).astype(np.int)
            self.model = nn.Sequential(nn.Conv2d(input_channel,
                                                 filter_num,
                                                 stride, stride=stride,
                                                 bias=False),
                                       nn.BatchNorm2d(filter_num, eps=1e-3, momentum=0.01),
                                       nn.ReLU())

    def forward(self, x):
        x = self.model(x)

        return x
