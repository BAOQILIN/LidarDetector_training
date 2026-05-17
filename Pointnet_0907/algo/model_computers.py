import os
from abc import ABC

import torch
import torch.onnx
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from ModelUtils.model_computers_base import model_computer_base
import networks


# class model_computer_base(metaclass=abc.ABCMeta):
#     """"父类中内容供参考,请勿增删改""""
#     def __init__(self, params_dict, result_root, res_dict={}):
#         self.params_dict = params_dict
#         self.result_root = result_root
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
#         return self.model(*inputs)
#
#     def load_model_params_torch(self, epoch, path=None, prefix=None, model=None):
#         if path is None:
#             path = os.path.join(os.path.join(self.result_root, self.params_dict['TRAIN']['PATH']['RESULT_PATH'][0]),
#                                 'model')
#         if prefix is None:
#             prefix = self.params_dict['TRAIN']['PATH']['SAVE_MODEL_PREFIX'][0]
#         # continue_epoch = self.params_dict['TRAIN']['CTRL']['CTRL_']['CONTINUE_EPOCH'][0]
#         # if continue_epoch == -1:
#         #     continue_epoch = None
#         # if continue_epoch is not None:
#         #     epoch += continue_epoch + 1
#         path = os.path.join(path, prefix + '_%d.torch' % (epoch))
#
#         if model is not None:
#             model.load_state_dict(torch.load(path))
#         else:
#             self.model.load_state_dict(torch.load(path))
#
#     def save_model_params_torch(self, epoch, model=None):
#         path = os.path.join(os.path.join(self.result_root, self.params_dict['TRAIN']['PATH']['RESULT_PATH'][0]),
#                             'model')
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
    def __init__(self, params_dict, result_root, res_dict={}):
        """
        函数头固定接口,不可增删改;函数体不可删改,可自定义增加内容
        params_dict: 解析algo_config.yaml文件中['TRAIN_MODEL']模块对应的参数字典
        result_root: 保存模型的onnx文件以及bin文件的根目录,具体保存文件夹路径os.path.join(result_root, params_dict['SAVE']['SAVE_PATH'][0])
        res_dict: res_dict['msg']是一个列表,列表中的每个元素均为字符串,用户通过向此列表中添加字符串,在前端页面打印相应的消息
        """
        super().__init__(params_dict, result_root, res_dict={})

    def _initial(self, epoch=None):
        """函数头和函数体不可增删改"""
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
        函数内容可自定义,但函数头不可增删改
        功能：保存指定epoch的onnx模型文件到本地,可在如下语句后续写具体的保存过程
        """
        save_model = self._initial(epoch)

        save_model_path = os.path.join(self.result_root, self.params_dict['SAVE']['SAVE_PATH'][0],
                                       self.params_dict['TRAIN']['PATH']['SAVE_MODEL_PREFIX'][0] + "_%s" % str(epoch).zfill(3))
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)

        # save input mlp
        input_onnx = onnx_feature_extrack(save_model,
                                          self.params_dict['TRAIN']['MODEL']['STRUCTURE']['INPUT_DIM'][0])
        inputs = Variable(torch.randn(1, 4)).cuda()
        onnx_path = os.path.join(save_model_path, 'input.onnx')
        name = ['xyz']
        dynamic_axes = {'xyz': {0: 'point_num'}}
        torch.onnx.export(input_onnx, inputs, onnx_path, verbose=True, input_names=name, dynamic_axes=dynamic_axes,
                          opset_version=11)

        # save output mlp: heading and type
        input_onnx = onnx_class_output_pointnet(save_model, self.params_dict['TRAIN']['MODEL']['STRUCTURE']['HEADING_BIN'][0])
        inputs = (Variable(torch.randn(4, self.params_dict['TRAIN']['MODEL']['STRUCTURE']['INPUT_LAYERS'][0][-1])).cuda(),
                  Variable(torch.randn(4, self.params_dict['TRAIN']['MODEL']['STRUCTURE']['XYZ_DIM'][0])).cuda())
        onnx_path = os.path.join(save_model_path, 'output_class.onnx')
        name = ['xyz_feature', 'center']
        dynamic_axes = {'xyz_feature': {0: 'batch_size'}, 'center': {0: 'batch_size'}}
        torch.onnx.export(input_onnx, inputs, onnx_path, verbose=True, input_names=name, dynamic_axes=dynamic_axes,
                          opset_version=11)

    def save_model_params_bin(self, epoch):
        """
        函数内容可自定义,但函数头不可增删改
        功能：保存指定epoch的bin模型文件到本地,可在如下语句后续写具体的保存过程
        """
        save_model = self._initial(epoch)

        save_model_path = os.path.join(self.result_root, self.params_dict['SAVE']['SAVE_PATH'],
                                       self.params_dict['TRAIN']['PATH']['SAVE_MODEL_PREFIX'][0] + "_%s" % str(epoch).zfill(3))
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)

        # save input mlp bin
        self._save_params_bin(save_model_path,
                              save_model,
                              len(self.params_dict['TRAIN']['MODEL']['STRUCTURE']['INPUT_LAYERS'][0]),
                              'input_mlp',
                              ['linear', 'batch_norm'],
                              ['feat_conv', 'feat_bn'])

        # save class layers
        self._save_params_bin(save_model_path,
                              save_model,
                              len(self.params_dict['TRAIN']['MODEL']['STRUCTURE']['CLASS_LAYERS'][0]),
                              'output_mlp.decode_mlp',
                              ['linear', 'batch_norm'],
                              ['fc', 'bn'])

        # save output layers
        self._save_params_bin_linear(save_model.state_dict()['output_mlp.output_mlp.0.weight'],
                                     save_model.state_dict()['output_mlp.output_mlp.0.bias'],
                                     os.path.join(save_model_path, 'head_reg.fc.bin'))

        self._save_params_bin_linear(save_model.state_dict()['output_mlp.output_mlp.1.weight'],
                                     save_model.state_dict()['output_mlp.output_mlp.1.bias'],
                                     os.path.join(save_model_path, 'head_cls.fc.bin'))

        self._save_params_bin_linear(save_model.state_dict()['output_mlp.output_mlp.2.weight'],
                                     save_model.state_dict()['output_mlp.output_mlp.2.bias'],
                                     os.path.join(save_model_path, 'type_cls.fc.bin'))


class onnx_feature_extrack(nn.Module, ABC):
    def __init__(self, model, input_dim):
        super(onnx_feature_extrack, self).__init__()
        
        self.input_onnx = model.model[0]
        self.input_dim = input_dim

    def forward(self, x):
        """
        Args:
            x: [P, C]
        Returns:
            x: [P, F]
        """
        x = x[:, :self.input_dim]
        x = self.input_onnx(x.unsqueeze(2)).squeeze(2)
        
        return x


class onnx_class_output_pointnet(nn.Module):
    def __init__(self, model, binNum):
        super(onnx_class_output_pointnet, self).__init__()
        
        self.output_onnx = model.model[1]
            
        self.pi = 3.1415927
        self.binNum = binNum

    def forward(self, x, center):
        """
        Args:
            x: [N, F]
            center: [N, D]
        Returns:
            heading reg: [N, 1]
            heading cls: softmax result, [N, heading_bin]
            type cls: softmax result, [N, type number]
        """
        outputs = self.output_onnx(x)
        
        outputs[0] = outputs[0].squeeze(1)
        outputs[1] = torch.argmax(outputs[1], dim = 1)
        outputs[2] = F.softmax(outputs[2], dim = 1)
        
        tmp = outputs[1]%2
        outputs[0] = 2 * self.pi / self.binNum * (outputs[1] + tmp) + torch.where(tmp > 0, -outputs[0], outputs[0])
        
        theta = torch.atan(center[:, 1]/center[:, 0])
        theta = torch.where((center[:, 0] < 0) & (center[:, 1] < 0), theta - self.pi, theta)
        theta = torch.where((center[:, 0] < 0) & (center[:, 1] > 0), theta + self.pi, theta)
        
        outputs[0] += theta
         
        return outputs[2], outputs[0].unsqueeze(1)


#     def func0(self, arg0, arg1, ...):
#         "示例：自定义类内方法，可删，可增加"
#
#
# def function0(arg0, arg1, ...):
#     "自定义类外函数，可删，可增加"
# 
# 
# class self_defined_cls():
#     "自定义类,可增删改"
    