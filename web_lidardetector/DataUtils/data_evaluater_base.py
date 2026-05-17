import os
import abc
import numpy as np


class data_evaluater_base(metaclass=abc.ABCMeta):
    def __init__(self, params_dict, result_root, train_flag, res_dict={}):
        self.params_dict = params_dict
        self.result_root = result_root
        self.train_flag = train_flag
        self.res_dict = res_dict
        if 'msg' not in self.res_dict:
            self.res_dict['msg'] = []

        self.result_path = os.path.join(self.result_root, self.params_dict['TRAIN']['PATH']['RESULT_PATH'][0])
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        self.evaluate_results = []

        if self.train_flag:
            self.evaluate_file = 'eva_vali.npy'
        else:
            self.evaluate_file = 'eva_test.npy'

        if os.path.exists(os.path.join(self.result_path, self.evaluate_file)):
            self.evaluate_results = np.load(os.path.join(self.result_path, self.evaluate_file),
                                                                allow_pickle=True).item()['evaluate_result']
        if self.params_dict['TRAIN']['OVERALL']['INITIAL_RESULT'][0]:
            self.evaluate_results = []

    def save(self):
        np.save(os.path.join(self.result_path, self.evaluate_file), {'evaluate_result': self.evaluate_results})

    @abc.abstractmethod
    def initial(self):
        pass

    @abc.abstractmethod
    def record(self, model_outputs, true):
        pass

    @abc.abstractmethod
    def evaluate(self):
        pass
