import torch
import torch.nn as nn
import numpy as np


class Head(nn.Module):
    def __init__(self, input_channel, filter_num, num_class=1, num_anchors_per_location=2):
        super(Head, self).__init__()
        self.num_anchors_per_location = num_anchors_per_location
        self.num_class = num_class
        self.code_size = 8
        self.conv_cls = nn.Sequential(nn.Conv2d(input_channel, filter_num,
                                                kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.BatchNorm2d(filter_num),
                                      nn.ReLU(),
                                      nn.Conv2d(filter_num, num_class * num_anchors_per_location,
                                                kernel_size=3, stride=1, padding=1))
        # self.conv_cls.cuda()

        self.conv_box = nn.ModuleDict()
        self.conv_box_names = []
        reg_config = {'reg': 2, 'height': 1, 'size': 3, 'angle': 2}
        for k, v in reg_config.items():
            cur_conv_list = [nn.Conv2d(input_channel, filter_num, kernel_size=3, stride=1, padding=1, bias=False),
                             nn.BatchNorm2d(filter_num),
                             nn.ReLU(),
                             nn.Conv2d(filter_num, num_anchors_per_location * v, kernel_size=3, stride=1, padding=1, bias=False)]
            self.conv_box[f'conv_{k}'] = nn.Sequential(*cur_conv_list)
            # self.conv_box.cuda()
            self.conv_box_names.append(f'conv_{k}')

        for m in self.conv_box.modules():
            if isinstance(m, nn.Conv2d):
                # 对二维卷积的权重参数采用kaiming正态分布
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.init_weights()

    def init_weights(self):
        pi = 0.01
        if isinstance(self.conv_cls, nn.Conv2d):  # False
            nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        else:  # True
            nn.init.constant_(self.conv_cls[-1].bias, -np.log((1 - pi) / pi))

    def forward(self, x):
        res_dict = {}

        cls_preds = self.conv_cls(x)
        box_preds = torch.cat([self.conv_box[reg_name](x) for reg_name in self.conv_box_names], dim=1)
        h, w = box_preds.shape[2:]
        batch_size = box_preds.shape[0]
        box_preds = box_preds.view(-1, self.num_anchors_per_location,
                                   # (b, 16, 128, 128)-->(b, 2, 8, 128, 128)-->(b, 2, 128, 128, 8)
                                   self.code_size, h, w).permute(0, 1, 3, 4, 2).contiguous()
        cls_preds = cls_preds.view(-1, self.num_anchors_per_location,
                                   # (b, 2*C', 128, 128)-->(b, 2, 1, 128, 128)-->(b, 2, 128, 128, 1)
                                   self.num_class, h, w).permute(0, 1, 3, 4, 2).contiguous()
        box_preds = box_preds.view(batch_size, -1, self.code_size)  # (b, 128*128*2, 8)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)  # (b, 128*128*2, 1)

        res_dict['cls_preds'] = cls_preds
        res_dict['box_preds'] = box_preds

        return res_dict



