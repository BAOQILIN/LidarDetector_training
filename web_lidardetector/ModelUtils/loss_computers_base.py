import abc
import os
import numpy as np


class loss_computer_base(metaclass=abc.ABCMeta):
    def __init__(self, params_dict, result_root, res_dict={}):
        self.params_dict = params_dict
        self.result_root = result_root
        self.res_dict = res_dict
        if 'msg' not in self.res_dict:
            self.res_dict['msg'] = []

        self.result_path = os.path.join(self.result_root, self.params_dict['TRAIN']['PATH']['RESULT_PATH'][0])
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        if not os.path.exists(os.path.join(self.result_path, 'losses.npy')):
            self.loss_record = []
        else:
            self.loss_record = self.load()

        if self.params_dict['TRAIN']['OVERALL']['INITIAL_RESULT'][0]:
            self.loss_record = []

    @abc.abstractmethod
    def loss_compute(self, outputs, labels, record=False):
        pass

    @abc.abstractmethod
    def _loss_initial(self):
        pass

    def load(self):
        return np.load(os.path.join(self.result_path, 'losses.npy'), allow_pickle=True).item()['loss_record']

    def save(self):
        np.save(os.path.join(self.result_path, 'losses.npy'), {'loss_record': self.loss_record})

    def initial(self):
        self.loss_record = []