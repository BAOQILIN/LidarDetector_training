import torch.nn as nn
import yaml
import sys
import os
sys.path.append(os.path.dirname(__file__))


class Network(nn.Module):
    def __init__(self, params_dict):
        """
        根据传入的参数字典和layer文件夹中的网络模块,解析得到完整的网络结构
        param params_dict: 解析network.yaml文件得到的参数字典
        """
        super(Network, self).__init__()
        
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

        # 示例
        # x = self.model[0](x);
        # output = self.model[1](torch.max(x, 2)[0])
        # return output

        output = 
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


# if __name__ == "__main__":
#     get_net()
