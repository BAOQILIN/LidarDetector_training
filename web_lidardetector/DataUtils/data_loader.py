import os
import abc
import copy
import numpy as np
import data_dataset
from importlib import reload


# class data_loader_base(metaclass=abc.ABCMeta):
    # def __init__(self, params_dict, data_root, train_flag, test_flag, res_dict={}):
    #     self.params_dict = params_dict
    #     self.data_root = data_root
    #     self.train_flag = train_flag
    #     self.test_flag = test_flag
    #     self.res_dict = res_dict
    #     if 'msg' not in self.res_dict:
    #         self.res_dict['msg'] = []

    #     self.dataset = {}
    #     self.dataset_index = {}
    #     self.dataset_length = {}
    #     self.dataset_index_i = {}

    #     if train_flag:  # 模型训练_验证过程
    #         #训练集
    #         param_dict_train = copy.deepcopy(self.params_dict)
    #         self.dataset['train'] = self._dataset_initial(self.params_dict['TRAIN']['OVERALL']['TRAIN_PREFIX'][0],
    #                                                       param_dict_train, self.data_root)
    #         self.dataset_length['train'] = len(self.dataset['train'])
    #         self.dataset_index['train'] = np.arange(self.dataset_length['train'])
    #         self.dataset_index_i['train'] = 0

    #         #验证集
    #         params_dict_validation = copy.deepcopy(self.params_dict)
    #         dataset_prefix = self.params_dict['VALIDATION']['VALI_']['PREFIX'][0]
    #         self.dataset['eva_vali'] = self._dataset_initial(dataset_prefix, params_dict_validation, self.data_root)
    #         self.dataset_length['eva_vali'] = len(self.dataset['eva_vali'])
    #         self.dataset_index['eva_vali'] = np.arange(self.dataset_length['eva_vali'])
    #         self.dataset_index_i['eva_vali'] = 0

    #     elif test_flag:  # 模型测试过程
    #         #测试集
    #         dataset_prefix = self.params_dict['TEST']['TEST_']['PREFIX'][0]
    #         self.dataset['eva_test'] = self._dataset_initial(dataset_prefix, self.params_dict, self.data_root)
    #         self.dataset_length['eva_test'] = len(self.dataset['eva_test'])
    #         self.dataset_index['eva_test'] = np.arange(self.dataset_length['eva_test'])
    #         self.dataset_index_i['eva_test'] = 0
            
    #     else:  # 模型感知推理过程
    #         dataset_prefix = self.params_dict['PREDICT']['PREDICT_']['PREFIX'][0]
    #         self.dataset['eva_predict'] = self._dataset_initial(dataset_prefix, self.params_dict, self.data_root)
    #         self.dataset_length['eva_predict'] = len(self.dataset['eva_predict'])
    #         self.dataset_index['eva_predict'] = np.arange(self.dataset_length['eva_predict'])
    #         self.dataset_index_i['eva_predict'] = 0

    # def initial(self, dataset_type):
    #     self.iter_dataset_type = dataset_type
    #     np.random.shuffle(self.dataset_index[dataset_type])
    #     self.iter_index = self.dataset_index[dataset_type]
    #     self.iter_index_i = 0

    # @abc.abstractmethod
    # def _dataset_initial(self, dataset_prefix, params_dict, data_root):
    #     pass

    # def __iter__(self):
    #     return self

    # def get_dataset(self, dataset_prefix, original_data=False):
      
    #     return self._dataset_initial(dataset_prefix, self.params_dict, self.data_root)


# class data_loader(metaclass=abc.ABCMeta):
class data_loader(object):
    def __init__(self, params_dict, data_root, train_flag, test_flag, res_dict={}):
        # super().__init__(params_dict, data_root, train_flag, test_flag, res_dict=res_dict)
        self.params_dict = params_dict
        self.data_root = data_root
        self.train_flag = train_flag
        self.test_flag = test_flag
        self.res_dict = res_dict
        if 'msg' not in self.res_dict:
            self.res_dict['msg'] = []

        self.dataset = {}
        self.dataset_index = {}
        self.dataset_length = {}
        self.dataset_index_i = {}

        if train_flag:  # 模型训练_验证过程
            #训练集
            param_dict_train = copy.deepcopy(self.params_dict)
            self.dataset['train'] = self._dataset_initial(self.params_dict['TRAIN']['OVERALL']['TRAIN_PREFIX'][0],
                                                          param_dict_train, self.data_root)
            self.dataset_length['train'] = len(self.dataset['train'])
            self.dataset_index['train'] = np.arange(self.dataset_length['train'])
            self.dataset_index_i['train'] = 0

            #验证集
            params_dict_validation = copy.deepcopy(self.params_dict)
            dataset_prefix = self.params_dict['VALIDATION']['VALI_']['PREFIX'][0]
            self.dataset['eva_vali'] = self._dataset_initial(dataset_prefix, params_dict_validation, self.data_root)
            self.dataset_length['eva_vali'] = len(self.dataset['eva_vali'])
            self.dataset_index['eva_vali'] = np.arange(self.dataset_length['eva_vali'])
            self.dataset_index_i['eva_vali'] = 0

        elif test_flag:  # 模型测试过程
            #测试集
            dataset_prefix = self.params_dict['TEST']['TEST_']['PREFIX'][0]
            self.dataset['eva_test'] = self._dataset_initial(dataset_prefix, self.params_dict, self.data_root)
            self.dataset_length['eva_test'] = len(self.dataset['eva_test'])
            self.dataset_index['eva_test'] = np.arange(self.dataset_length['eva_test'])
            self.dataset_index_i['eva_test'] = 0
            
        else:  # 模型感知推理过程
            dataset_prefix = self.params_dict['PREDICT']['PREDICT_']['PREFIX'][0]
            self.dataset['eva_predict'] = self._dataset_initial(dataset_prefix, self.params_dict, self.data_root)
            self.dataset_length['eva_predict'] = len(self.dataset['eva_predict'])
            self.dataset_index['eva_predict'] = np.arange(self.dataset_length['eva_predict'])
            self.dataset_index_i['eva_predict'] = 0

    def initial(self, dataset_type):
        self.iter_dataset_type = dataset_type
        np.random.shuffle(self.dataset_index[dataset_type])
        self.iter_index = self.dataset_index[dataset_type]
        self.iter_index_i = 0

    def _dataset_initial(self, dataset_prefix, params_dict, data_root):
        return data_dataset.dataset(params_dict, os.path.join(data_root, params_dict['TRAIN']['PATH']['DATASET_PATH'][0]), dataset_prefix, self.res_dict)

    def get_dataset(self, dataset_prefix, original_data=False):
        return self._dataset_initial(dataset_prefix, self.params_dict, self.data_root)

    def __iter__(self):
        return self

    def __next__(self):
        self.dataset_length[self.iter_dataset_type] = self.dataset[self.iter_dataset_type].num_batch

        inputs, label, filenames = self.dataset[self.iter_dataset_type][self.iter_index_i]

        if inputs is None or (label is None and self.train_flag):
            raise StopIteration

        self.iter_index_i += 1

        return inputs, label, filenames, self.iter_index_i
