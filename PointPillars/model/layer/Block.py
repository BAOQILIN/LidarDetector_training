import torch.nn as nn


class Block(nn.Module):
    def __init__(self, layer_num, input_channel, filter_num, stride):
        super().__init__()
        model = [nn.Conv2d(input_channel,
                           filter_num,
                           kernel_size=3,
                           stride=stride,
                           padding=1, bias=False,
                           padding_mode='zeros'),
                 nn.BatchNorm2d(filter_num, eps=1e-3, momentum=0.01),
                 nn.ReLU()]
        for i in range(layer_num):
            model.extend([nn.Conv2d(filter_num,
                                    filter_num,
                                    kernel_size=3,
                                    padding=1,
                                    bias=False),
                          nn.BatchNorm2d(filter_num, eps=1e-3, momentum=0.01),
                          nn.ReLU()])
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)

        return x
