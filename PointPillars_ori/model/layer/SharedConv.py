import torch
import torch.nn as nn


class SharedConv(nn.Module):
    def __init__(self, input_channel, filter_num):
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(input_channel,
                                             filter_num,
                                             kernel_size=3, stride=1,
                                             padding=1, bias=False),
                                   nn.BatchNorm2d(filter_num,
                                                  eps=1e-3,
                                                  momentum=0.01),
                                   nn.ReLU())
        # self.model.cuda()

    def forward(self, input):
        output = self.model(input)

        return output
