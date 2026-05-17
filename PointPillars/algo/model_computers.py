import os
import torch
import torch.onnx
import torch.nn as nn
from ModelUtils.model_computers_base import model_computer_base
import networks
import utils


# class model_computer_base(metaclass=abc.ABCMeta):
#     def __init__(self, params_dict, result_root, res_dict={}):
#         self.params_dict = params_dict
#         self.result_root = result_root
#         self.res_dict = res_dict
#         if 'msg' not in self.res_dict:
#             self.res_dict['msg'] = []

#         continue_epoch = self.params_dict['TRAIN']['CTRL']['CTRL_']['CONTINUE_EPOCH'][0]
#         if continue_epoch == -1:
#             continue_epoch = None
#         self.model = self._initial(continue_epoch)

#         result_path = os.path.join(self.result_root, self.params_dict['TRAIN']['PATH']['RESULT_PATH'][0])
#         if not os.path.exists(result_path):
#             os.makedirs(result_path)

#     #    @abc.abstractmethod
#     def save_model_params_onnx(self, epoch):
#         pass

#     #    @abc.abstractmethod
#     def save_model_params_bin(self, epoch):
#         pass

#     def model_freeze(self, freeze_layer_names):
#         self.model.freeze(freeze_layer_names)

#     def model_compute(self, inputs):
#         return self.model(inputs)

#     def load_model_params_torch(self, epoch, prefix=None, model=None):
#         path = os.path.join(self.result_root, 'model_torch')
#         if prefix is None:
#             prefix = self.params_dict['TRAIN']['PATH']['SAVE_MODEL_PREFIX'][0]
#         # continue_epoch = self.params_dict['TRAIN']['CTRL']['CTRL_']['CONTINUE_EPOCH'][0]
#         # if continue_epoch == -1:
#         #     continue_epoch = None
#         # if continue_epoch is not None:
#         #     epoch += continue_epoch + 1
#         path = os.path.join(path, prefix + '_%d.torch' % (epoch))

#         if model is not None:
#             model.load_state_dict(torch.load(path))
#         else:
#             self.model.load_state_dict(torch.load(path))

#     def save_model_params_torch(self, epoch, model=None):
#         path = os.path.join(self.result_root, 'model_torch')
#         if not os.path.exists(path):
#             os.makedirs(path)
#         path = os.path.join(path, self.params_dict['TRAIN']['PATH']['SAVE_MODEL_PREFIX'][0] + '_%d.torch' % (epoch))
#         if model is not None:
#             torch.save(model.state_dict(), path)
#         else:
#             torch.save(self.model.state_dict(), path)

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

#     def _save_params_bin_linear(self, weight, bias, path):
#         with open(path, 'wb') as f:
#             for data in weight.view(-1):
#                 f.write(struct.pack('d', data))

#             for data in bias.view(-1):
#                 f.write(struct.pack('d', data))

#     def _save_params_bin_batchnorm(self, weight, bias, running_mean, running_var, path):
#         with open(path, 'wb') as f:

#             for data in weight.view(-1):
#                 f.write(struct.pack('d', data))

#             for data in bias.view(-1):
#                 f.write(struct.pack('d', data))

#             for data in running_mean.view(-1):
#                 f.write(struct.pack('d', data))

#             for data in running_var.view(-1):
#                 f.write(struct.pack('d', data))


class model_computer(model_computer_base):
    def __init__(self, params_dict, model_epoch_root, res_dict={}):
        """
        函数头固定接口,不可增删改;函数体不可删改,可自定义增加内容
        params_dict: 解析algo_config.yaml中的['TRAIN_MODEL']部分得到的参数字典,具体内容用户可在该文件中自定义
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

        model_onnx_folder = os.path.join(self.model_epoch_root,
                                         self.params_dict['TRAIN']['PATH']['SAVE_MODEL_PREFIX'][0] + "_epoch_%d" % (epoch))
        if not os.path.exists(model_onnx_folder):
            os.makedirs( model_onnx_folder)

        # save_vfe
        vfe_pfnlayer_onnx = OnnxVfePfnlayer(save_model.VFE, save_model.PFNLayers)
        vfe_pfnlayer_onnx.cuda()
        vfe_pfnlayer_onnx.eval()

        voxel_features = torch.ones(2, 20, 4).float().cuda()
        voxel_coords = torch.ones(2, 4).float().cuda()
        point_num_per_voxel = torch.ones(2).float().cuda()
        input_list = (voxel_features, point_num_per_voxel, voxel_coords)
        input_names = ['voxel_features', 'point_num_per_voxel', 'voxel_coords']
        dynamic_axes = {'voxel_features': {0: 'pillar_num'},
                        'point_num_per_voxel': {0: 'pillar_num'},
                        'voxel_coords': {0: 'pillar_num'}}
        vfe_onnx_path = os.path.join(model_onnx_folder, 'vfe.onnx')

        torch.onnx.export(vfe_pfnlayer_onnx, input_list, vfe_onnx_path, verbose=True,
                          input_names=input_names, dynamic_axes=dynamic_axes, opset_version=11)

        # save_backbone
        backbone_onnx = OnnxBackbone(save_model.Block1, save_model.Block2, save_model.Block3,
                                     save_model.Deblock1, save_model.Deblock2, save_model.Deblock3)
        backbone_onnx.cuda()
        backbone_onnx.eval()
        spatial_features = torch.ones(1, 64, 1024, 512).float().cuda()
        input_list = spatial_features
        input_names = ['spatial_features']
        backbone_onnx_path = os.path.join(model_onnx_folder, 'backbone2D.onnx')
        torch.onnx.export(backbone_onnx, input_list, backbone_onnx_path, verbose=True, input_names=input_names,
                          opset_version=11)

        # save_rpn
        rpn_onnx = OnnxRPN(save_model.SharedConv, save_model.Head1, save_model.Head2,
                           save_model.Head3, save_model.Head4, save_model.Head5, save_model.Merge)
        rpn_onnx.cuda()
        rpn_onnx.eval()
        spatial_features_2d = torch.ones(1, 384, 256, 128).float().cuda()
        input_list = spatial_features_2d
        input_names = ['spatial_features_2d']
        output_names = ['batch_cls_preds', 'batch_box_preds']
        rpn_onnx_path = os.path.join(model_onnx_folder, 'rpn.onnx')
        torch.onnx.export(rpn_onnx, input_list, rpn_onnx_path, verbose=True,
                          input_names=input_names, output_names=output_names, opset_version=11)

    def save_model_params_bin(self, epoch):
            """
            函数头不可增删改,函数体不可删改,但需要增加并完善具体的保存过程
            功能：保存指定epoch的bin模型文件到本地,可在如下语句后续写具体的保存过程
            """
            save_model = self._initial(epoch)

            save_model_path = os.path.join(self.model_epoch_root,
                                           self.params_dict['TRAIN']['PATH']['SAVE_MODEL_PREFIX'][0] + "_epoch_%d" % (epoch))
            if not os.path.exists(save_model_path):
                os.system("mkdir -p " + save_model_path)


class OnnxVfePfnlayer(nn.Module):
    def __init__(self, vfe, PFNLayers):
        super(OnnxVfePfnlayer, self).__init__()
        self.vfe = vfe
        self.PFNLayers = PFNLayers

    def forward(self, voxel, num_pts_voxel, voxel_coors):
        ret = self.vfe(voxel, num_pts_voxel, voxel_coors)
        print('ret.shape: ', ret.shape)
        output = self.PFNLayers(ret)

        return output


class OnnxBackbone(nn.Module):
    def __init__(self, block1, block2, block3, deblock1, deblock2, deblock3):
        super(OnnxBackbone, self).__init__()
        self.block1 = block1
        self.block2 = block2
        self.block3 = block3
        self.deblock1 = deblock1
        self.deblock2 = deblock2
        self.deblock3 = deblock3

    def forward(self, spatial_features):
        upsamples = []
        x = self.block1(spatial_features)
        upsamples.append(self.deblock1(x))
        x = self.block2(x)
        upsamples.append(self.deblock2(x))
        x = self.block3(x)
        upsamples.append(self.deblock3(x))
        output = torch.cat(upsamples, dim=1)

        return output


class OnnxRPN(nn.Module):
    def __init__(self, sharedconv, head1, head2, head3, head4, head5, merge):
        super(OnnxRPN, self).__init__()
        self.sharedconv = sharedconv
        self.head1 = head1
        self.head2 = head2
        self.head3 = head3
        self.head4 = head4
        self.head5 = head5
        self.merge = merge

    def forward(self, spatial_feature_2d):
        x = self.sharedconv(spatial_feature_2d)
        res1 = self.head1(x)
        res2 = self.head2(x)
        res3 = self.head3(x)
        res4 = self.head4(x)
        res5 = self.head5(x)
        res6 = self.merge(res1, res2, res3, res4, res5)
        batch_cls_preds = torch.cat(res6['cls_preds'], dim=1)
        batch_box_preds = torch.cat(res6['box_preds'], dim=1)
        output = [batch_cls_preds, batch_box_preds]

        return output

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
