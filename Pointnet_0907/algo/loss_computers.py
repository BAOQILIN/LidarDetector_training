import os.path
import numpy as np
import torch
import torch.nn.functional as F
from ModelUtils.loss_computers_base import loss_computer_base
import sys
sys.path.insert(0, os.path.dirname(__file__))


# class loss_computer_base(metaclass=abc.ABCMeta):
#     """父类中内容供参考,请勿增删改"""
#     def __init__(self, params_dict, result_root, res_dict={}):
#         self.params_dict = params_dict
#         self.result_root = result_root
#         self.res_dict = res_dict
#         if 'msg' not in self.res_dict:
#             self.res_dict['msg'] = []
#
#         result_path = os.path.join(self.result_root, self.params_dict['TRAIN']['PATH']['RESULT_PATH'][0])
#         if not os.path.exists(result_path):
#             os.makedirs(result_path)
#
#         if not os.path.exists(os.path.join(result_path, 'losses.npy')):
#             self.loss_record = {}
#         else:
#             self.loss_record = self.load()
#
#         if self.params_dict['TRAIN']['OVERALL']['INITIAL_RESULT'][0]:
#             self.loss_record = []
#
#     #    @abc.abstractmethod
#     def loss_compute(self, outputs, labels, record=False):
#         pass
#
#     #    @abc.abstractmethod
#     def _loss_initial(self):
#         pass
#
#     def load(self):
#         result_path = os.path.join(self.result_root, self.params_dict['TRAIN']['PATH']['RESULT_PATH'][0])
#         return np.load(os.path.join(result_path, 'losses.npy'), allow_pickle=True).item()
#
#     def save(self):
#         result_path = os.path.join(self.result_root, self.params_dict['TRAIN']['PATH']['RESULT_PATH'][0])
#         np.save(os.path.join(result_path, 'losses.npy'), self.loss_record)
#
#     def initial(self):
#         self.loss_record = []


class loss_computer(loss_computer_base):
    def __init__(self, params_dict, result_root, res_dict={}):
        """
        函数头固定接头,不可增删改;函数体不可删改,可自定义增加内容
        params_dict: algo_config.yaml中 TRAIN_MODEL模块对应的参数字典
        result_root: 保存损失结果文件losses.npy的根目录,具体的保存路径见loss_computer_base.save()方法
                     关于result_root的文件夹结构,参见《文件层级结构.txt》
        res_dict: res_dict['msg']是一个列表,列表中的每个元素均为字符串,用户通过向此列表中添加字符串,在web前端页面打印相应的消息
        """
        super().__init__(params_dict, result_root, res_dict)
        self._loss_initial()

    def _loss_initial(self):
        """
        函数头固定接口，不可增删改;函数体自定义
        以列表形式保存每个batch中生成的loss,可以根据loss类型自定义,
        在loss_compute()方法中,可以自行对此处的损失保存到self.loss_record变量中,
        框架会自动调用loss_computer_base.save()方法保存到指定文件losses.npy中
            eg: self.box_losses = [] 保存预测框的回归损失
                self.cls_losses = [] 保存预测框的类别损失
                self.total_loss = [] 总损失
        """
        self.heading_reg_loss = []
        self.heading_dir_loss = []
        self.class_loss = []
        self.total_loss = []

    def loss_compute(self, model_outputs, labels, record=False):
        """
        固定接口，不可删除
        功能: 计算单个batch的 model_outputs 和 label之间的损失值
        :param model_outputs: 由model_computers.py中 class model_computer.model_compute()方法得到的单个batch模型预测值
                              同时也是networks.py文件中 class Network.forward()方法的返回值
        :param labels: 由data_loader.py中 class data_loader_XXX的 __next__()方法得到的单个batch 标签
        :param record: 根据当前batch在整个epoch中的索引和params_dict['TRAIN']['CTRL']['LOSS']['PRINT_GAP'][0]是否整除
                       确定是否保存__loss_init(self)方法中相关损失值
        :return: inputs和labels之间的总损失值，用于反向传播计算梯度、更新权重
        """

        # 自定义各种损失计算, 并将计算结果添加到__loss_init(self)方法中损失列表中去,例如:
        heading_mask = labels[-1]
        heading_reg_loss = F.smooth_l1_loss(model_outputs[0][heading_mask].view(-1), labels[0][heading_mask])
        heading_dir_loss = F.cross_entropy(model_outputs[1][heading_mask], labels[1][heading_mask])
        class_loss = torch.nn.CrossEntropyLoss()(model_outputs[2], labels[2])

        total_loss = int(self.params_dict['TRAIN']['CTRL']['LOSS']['WEIGHT'][0][0]) * heading_reg_loss + \
                     int(self.params_dict['TRAIN']['CTRL']['LOSS']['WEIGHT'][0][1]) * heading_dir_loss + \
                     int(self.params_dict['TRAIN']['CTRL']['LOSS']['WEIGHT'][0][2]) * class_loss
        self.heading_reg_loss.append(heading_reg_loss.item())
        self.heading_dir_loss.append(heading_dir_loss.item())
        self.class_loss.append(class_loss.item())
        self.total_loss.append(total_loss.item())

        if record:
            # 添加到self.loss_record中的值来自__loss_init(self)方法中定义的损失值列表
            # 父类save()方法保存self.loss_record变量至本地指定路径
            self.loss_record.append([np.mean(self.heading_reg_loss),
                                     np.mean(self.heading_dir_loss),
                                     np.mean(self.class_loss),
                                     np.mean(self.total_loss)])
            self.res_dict['msg'].append(self.loss_record[-1])

            self._loss_initial()

        return total_loss  # 返回的必须是batch数据的总损失值,且tensor.size=1,才能进行反向传播


#     def func0(self, arg0, arg1, ...):
#         "示例：自定义类内方法，可删，可增加"
#
#
# def function0(arg0, arg1, ...):
#     "自定义类外函数，可删，可增加"
