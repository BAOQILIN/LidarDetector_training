import os
import torch
import torch.onnx
from ModelUtils.model_computers_base import model_computer_base
import networks
import utils  # (公共文件)


# class model_computer_base(metaclass=abc.ABCMeta):
#     def __init__(self, params_dict, result_root, model_epoch_root, res_dict={}):
#         self.params_dict = params_dict
#         self.result_root = result_root
#         self.model_epoch_root = model_epoch_root
#         self.res_dict = res_dict
#         if 'msg' not in self.res_dict:
#             self.res_dict['msg'] = []
#
#         continue_epoch = self.params_dict['TRAIN']['CTRL']['CTRL_']['CONTINUE_EPOCH'][0]
#         if continue_epoch == -1:
#             continue_epoch = None
#         self.model = self._initial(continue_epoch)
#
#         result_path = os.path.join(self.result_root, self.params_dict['TRAIN']['PATH']['RESULT_PATH'][0])
#         if not os.path.exists(result_path):
#             os.makedirs(result_path)
#
#     #    @abc.abstractmethod
#     def save_model_params_onnx(self, epoch):
#         pass
#
#     #    @abc.abstractmethod
#     def save_model_params_bin(self, epoch):
#         pass
#
#     def model_freeze(self, freeze_layer_names):
#         self.model.freeze(freeze_layer_names)
#
#     def model_compute(self, inputs):
#         return self.model(inputs)
#
#     def load_model_params_torch(self, epoch, prefix=None, model=None):
#         if prefix is None:
#             prefix = self.params_dict['TRAIN']['PATH']['SAVE_MODEL_PREFIX'][0]
#         path = os.path.join(self.model_epoch_root, prefix + '_%d.torch' % (epoch))
#
#         if model is not None:
#             model.load_state_dict(torch.load(path))
#         else:
#             self.model.load_state_dict(torch.load(path))
#
#     def save_model_params_torch(self, epoch, model=None):
#         path = os.path.join(self.result_root, 'model_torch')
#         if not os.path.exists(path):
#             os.makedirs(path)
#         path = os.path.join(path, self.params_dict['TRAIN']['PATH']['SAVE_MODEL_PREFIX'][0] + '_%d.torch' % (epoch))
#         if model is not None:
#             torch.save(model.state_dict(), path)
#         else:
#             torch.save(self.model.state_dict(), path)
#
#     def _save_params_bin(self, save_path, save_model, layer_number, param_key, param_types, save_prefixs):
#         for i in range(layer_number):
#             base_name = param_key + '.layers.' + str(i)
#             for j in range(len(param_types)):
#                 path = os.path.join(save_path, save_prefixs[j] + str(i + 1) + '.bin')
#                 if param_types[j] == 'linear':
#                     self._save_params_bin_linear(save_model.state_dict()[base_name + '.linear.weight'],
#                                                  save_model.state_dict()[base_name + '.linear.bias'],
#                                                  path)
#                 if param_types[j] == 'batch_norm':
#                     self._save_params_bin_batchnorm(save_model.state_dict()[base_name + '.norm.weight'],
#                                                     save_model.state_dict()[base_name + '.norm.bias'],
#                                                     save_model.state_dict()[base_name + '.norm.running_mean'],
#                                                     save_model.state_dict()[base_name + '.norm.running_var'],
#                                                     path)
#
#     def _save_params_bin_linear(self, weight, bias, path):
#         with open(path, 'wb') as f:
#             for data in weight.view(-1):
#                 f.write(struct.pack('d', data))
#
#             for data in bias.view(-1):
#                 f.write(struct.pack('d', data))
#
#     def _save_params_bin_batchnorm(self, weight, bias, running_mean, running_var, path):
#         with open(path, 'wb') as f:
#
#             for data in weight.view(-1):
#                 f.write(struct.pack('d', data))
#
#             for data in bias.view(-1):
#                 f.write(struct.pack('d', data))
#
#             for data in running_mean.view(-1):
#                 f.write(struct.pack('d', data))
#
#             for data in running_var.view(-1):
#                 f.write(struct.pack('d', data))


class model_computer(model_computer_base):
    def __init__(self, params_dict, model_epoch_root, res_dict={}):
        """
        函数头固定接口,不可增删改;函数体不可删改,可自定义增加内容
        params_dict: 解析algo_config.yaml中的['TRAIN_MODEL']部分得到的参数字典,具体内容用户可在该文件中自定义
        model_epoch_root: 保存模型的根目录
        res_dict: res_dict['msg']是一个列表,列表中的每个元素均为字符串,用户通过向此列表中添加字符串,在前端页面打印相应的消息
        """
        super().__init__(params_dict, model_epoch_root, res_dict={})

    def _initial(self, epoch=None):
        """
        函数头和函数体不可增删改
        功能：加载指定epoch的模型
        """
        model = networks.get_net()

        if epoch is not None:
            self.load_model_params_torch(epoch,
                                         prefix=self.params_dict['TRAIN']['PATH']['CONTINUE_MODEL_PREFIX'][0],
                                         model=model)

        if torch.cuda.is_available():
            model = model.cuda()

        return model

    def save_model_params_onnx(self, epoch):
        """
        函数头不可增删改,函数体不可删改,但需要增加并完善具体的保存过程
        功能：保存指定epoch的onnx模型文件到本地,可在如下语句后续写具体的保存过程
        """
        save_model = self._initial(epoch)

        save_model_path = os.path.join(self.model_epoch_root,
                                       self.params_dict['TRAIN']['PATH']['SAVE_MODEL_PREFIX'][0] + "epoch_%d" % (epoch))
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)

        "具体实现代码"

    def save_model_params_bin(self, epoch):
        """
        函数头不可增删改,函数体不可删改,但需要增加并完善具体的保存过程
        功能：保存指定epoch的bin模型文件到本地,可在如下语句后续写具体的保存过程
        """
        save_model = self._initial(epoch)

        save_model_path = os.path.join(self.model_epoch_root, 
                                       self.params_dict['TRAIN']['PATH']['SAVE_MODEL_PREFIX'][0] + "epoch_%d" % (epoch))
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)

        "具体实现代码"

#     def func0(self, arg0, arg1, ...):
#         "示例:自定义类内方法"
#
#
# def function0(arg0, arg1, ...):
#     "示例:自定义类外函数"
# 
# 
# class self_defined_cls():
#     "示例:自定义类"
