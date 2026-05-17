import abc
import os
import torch
import struct


class model_computer_base(metaclass=abc.ABCMeta):
    def __init__(self, params_dict, model_epoch_root, pretrained_path=None, res_dict={}):
        self.params_dict = params_dict
        self.model_epoch_root = model_epoch_root
        self.pretrained_path = pretrained_path
        self.res_dict = res_dict
        if 'msg' not in self.res_dict:
            self.res_dict['msg'] = []

        # continue_epoch = self.params_dict['TRAIN']['CTRL']['CTRL_'].get('CONTINUE_EPOCH', [-1])[0]
        # if continue_epoch == -1:
        #     continue_epoch = None

        self.model = self._initial(self.pretrained_path)

    @abc.abstractmethod
    def save_model_params_onnx(self, epoch):
        pass

    @abc.abstractmethod
    def save_model_params_bin(self, epoch):
        pass

    def model_freeze(self, freeze_layer_names):
        self.model.freeze(freeze_layer_names)

    def model_compute(self, inputs):
        return self.model(inputs)

    def load_model_params_torch(self, epoch, prefix=None, model=None):
        if str(epoch).isdigit():
            if prefix is None:
                prefix = self.params_dict['TRAIN']['PATH']['SAVE_MODEL_PREFIX'][0]
            path = os.path.join(self.model_epoch_root, prefix + '_%d.torch' % (epoch))
        else:
            path = epoch

        if not os.path.exists(path):
            raise FileNotFoundError('Pretrained File: ' + path + ' Not Found.')

        if model is not None:
            model.load_state_dict(torch.load(path))
        else:
            self.model.load_state_dict(torch.load(path))

    def save_model_params_torch(self, epoch, model=None):
        if not os.path.exists(self.model_epoch_root):
            os.makedirs(self.model_epoch_root)
        path = os.path.join(self.model_epoch_root, self.params_dict['TRAIN']['PATH']['SAVE_MODEL_PREFIX'][0] + '_%d.torch' % (epoch))
        if model is not None:
            torch.save(model.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)