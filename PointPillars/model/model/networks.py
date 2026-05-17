import torch.nn as nn
import yaml
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from layer.VFE import VFE
from layer.PFNLayers import PFNLayers
from layer.Scatter import Scatter
from layer.Block import Block
from layer.Deblock import Deblock
from layer.SharedConv import SharedConv
from layer.Head import Head
from layer.Merge import Merge


class Network(nn.Module):
    def __init__(self, params_dict):
        """
        根据传入的参数字典和layer文件夹中的网络模块,解析得到完整的网络结构
        param params_dict: 解析network.yaml文件得到的参数字典
        """
        super(Network, self).__init__()
        self.params_dict = params_dict

        for i in range(len(self.params_dict) - 1):
            cls_name, _, from_, args = self.params_dict[i]
            if -1 < i < 3:
                exec(f'self.{cls_name} = {cls_name}(*args)')
            elif 2 < i < 6:
                exec(f'self.{cls_name}{i-2} = {cls_name}(*args)')
            elif 5 < i < 9:
                exec(f'self.{cls_name}{i-5} = {cls_name}(*args)')
            elif i == 9:
                exec(f'self.{cls_name} = {cls_name}(*args)')
            elif 9 < i < 15:
                exec(f'self.{cls_name}{i-9} = {cls_name}(*args)')
            else:
                exec(f'self.{cls_name} = {cls_name}(*args)')

        # 示例
        # self.block = None
        # layers = []
        # for i in range(len(params_dict)):
        #     inputs_, cls_name, args = params_dict[i]
        #     exec('self.block = ' + cls_name + '(*args)')
        #     layers.append(self.block)
        # self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        函数名不可改变,形参列表自定义,前向传播
        param x: 输入模型的单个batch数据,即data_dataset.py文件中class dataset.__getitem__()方法的返回值inputs
        return: 模型推理计算结果
                 传参到 data_evaluater.py文件中 class data_evaluater.record()方法
                 传参到 loss_computers.py文件中 class loss_computer.loss_compute()方法
        """
        voxel, num_pts_voxel, voxel_coors = x
        x = self.VFE(voxel, num_pts_voxel, voxel_coors)
        x = self.PFNLayers(x)
        x = self.Scatter(x, voxel_coors)

        upsamples = []
        x = self.Block1(x)
        upsamples.append(self.Deblock1(x))
        x = self.Block2(x)
        upsamples.append(self.Deblock2(x))
        x = self.Block3(x)
        upsamples.append(self.Deblock3(x))
        x = torch.cat(upsamples, dim=1)

        x = self.SharedConv(x)
        res1 = self.Head1(x)
        res2 = self.Head2(x)
        res3 = self.Head3(x)
        res4 = self.Head4(x)
        res5 = self.Head5(x)
        output = self.Merge(res1, res2, res3, res4, res5)

        return output

def get_net():
    """
    固定接口,函数头与函数体均不可增删改,由model_computers.py文件中的 class model_computer._initial()方法调用
    :return: 网络模型
    """
    cfg_path = os.path.join(os.path.dirname(__file__), 'networks.yaml')
    with open(cfg_path, encoding='utf-8') as f:
        params_dict = yaml.safe_load(f)
    model = Network(params_dict)
    return model


if __name__ == "__main__":
    get_net()
