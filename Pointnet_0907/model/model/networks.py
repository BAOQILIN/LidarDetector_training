import torch.nn as nn
import yaml
import torch
import sys
import os
sys.path.append(os.path.dirname(__file__))
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, params_dict):
        """
        函数头
        :param params_dict:
        """
        super(Network, self).__init__()
        self.mlp = MLP(params_dict)
        self.decoder_pointnet = Decoder_pointnet(params_dict)

        self.model = nn.Sequential(self.mlp, self.decoder_pointnet)

    def forward(self, x):
        """
        函数名不可改变,形参列表自定义,前向传播
        param x: 输入模型的单个batch数据,即data_dataset.py文件中class dataset.__getitem__()方法的返回值inputs
        return: 模型推理计算结果
                 传参到 data_evaluater.py文件中 class data_evaluater.record()方法
                 传参到 loss_computers.py文件中 class loss_computer.loss_compute()方法
        """
        x = x[0]
        x = self.model[0](x.transpose(2, 1))
        x = torch.max(x, 2)[0]
        x = self.model[1](x)

        return x


class MLP(nn.Module):
    def __init__(self, params_dict):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(0, 7):
            cls_name, model_type, inputs_, args = params_dict[i]
            if model_type == 'normal':
                exec('self.block1 = ' + cls_name + '(*args)')
            elif model_type == 'pytorch':
                exec('self.block1 = nn.' + cls_name + '(*args)')
            self.layers.append(self.block1)
        del self.block1

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class Decoder_pointnet(nn.Module):
    def __init__(self, params_dict):
        super(Decoder_pointnet, self).__init__()

        self.decode_mlp = nn.ModuleList()
        for i in range(8, 16):
            cls_name, model_type, inputs_, args = params_dict[i]
            if model_type == 'normal':
                exec('self.block2 = ' + cls_name + '(*args)')
            elif model_type == 'pytorch':
                exec('self.block2 = nn.' + cls_name + '(*args)')
            self.decode_mlp.append(self.block2)
        del self.block2

        self.output_mlp = nn.ModuleList()
        for i in range(16, 19):
            cls_name, model_type, inputs_, args = params_dict[i]
            if model_type == 'normal':
                exec('self.block3 = ' + cls_name + '(*args)')
            elif model_type == 'pytorch':
                exec('self.block3 = nn.' + cls_name + '(*args)')
            self.output_mlp.append(self.block3)
        del self.block3

    def forward(self, x):
        x = x.unsqueeze(2)
        for layer in self.decode_mlp:
            x = layer(x)
        x = x.squeeze(2)
        F.relu(x, inplace=True)

        output = []
        for i in range(3):
            output.append(self.output_mlp[i](x))

        return output


def get_net():
    """
    固定接口,函数头与函数体均不可增删改,由model_computers.py文件中的 class model_computer的_initial()方法调用
    :return: 网络模型
    """
    cfg_path = os.path.join(os.path.dirname(__file__), 'networks.yaml')
    with open(cfg_path, encoding='utf-8') as f:
        params_dict = yaml.safe_load(f)
    model = Network(params_dict)
    return model

