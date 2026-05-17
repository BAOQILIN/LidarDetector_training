import copy
import os
import sys

import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'PointPillars', 'algo'))
import data_dataset


class data_loader(object):
    def __init__(self, params_dict, data_root, train_flag, test_flag, res_dict={}):
        self.params_dict = params_dict
        self.data_root = data_root
        self.train_flag = train_flag
        self.test_flag = test_flag
        self.res_dict = res_dict
        if 'msg' not in self.res_dict:
            self.res_dict['msg'] = []

        self.dataset = {}
        self.dataset_length = {}
        self.loaders = {}
        self.iter_loader = None
        self.iter_dataset_type = None

        if train_flag:
            param_dict_train = copy.deepcopy(self.params_dict)
            self.dataset['train'] = self._dataset_initial(self.params_dict['TRAIN']['OVERALL']['TRAIN_PREFIX'][0], param_dict_train, self.data_root)
            self.dataset_length['train'] = len(self.dataset['train'])

            params_dict_validation = copy.deepcopy(self.params_dict)
            dataset_prefix = self.params_dict['VALIDATION']['VALI_']['PREFIX'][0]
            self.dataset['eva_vali'] = self._dataset_initial(dataset_prefix, params_dict_validation, self.data_root)
            self.dataset_length['eva_vali'] = len(self.dataset['eva_vali'])
        elif test_flag:
            dataset_prefix = self.params_dict['TEST']['TEST_']['PREFIX'][0]
            self.dataset['eva_test'] = self._dataset_initial(dataset_prefix, self.params_dict, self.data_root)
            self.dataset_length['eva_test'] = len(self.dataset['eva_test'])
        else:
            dataset_prefix = self.params_dict['PREDICT']['PREDICT_']['PREFIX'][0]
            self.dataset['eva_predict'] = self._dataset_initial(dataset_prefix, self.params_dict, self.data_root)
            self.dataset_length['eva_predict'] = len(self.dataset['eva_predict'])

        self._build_loaders()

    def _build_loaders(self):
        data_cfg = self.params_dict['TRAIN']['CTRL']['DATA']
        num_workers = data_cfg.get('NUM_WORKERS', [0])[0]
        pin_memory = data_cfg.get('PIN_MEMORY', [False])[0]
        persistent_workers = data_cfg.get('PERSISTENT_WORKERS', [False])[0]
        prefetch_factor = data_cfg.get('PREFETCH_FACTOR', [2])[0]
        batch_size = data_cfg['BATCH_SIZE'][0]

        for dataset_type, dataset_obj in self.dataset.items():
            shuffle = dataset_type == 'train'
            loader_kwargs = {
                'dataset': dataset_obj,
                'batch_size': batch_size,
                'shuffle': shuffle,
                'num_workers': num_workers,
                'pin_memory': pin_memory,
                'drop_last': False,
                'collate_fn': dataset_obj.collate_batch,
            }
            if num_workers > 0:
                loader_kwargs['persistent_workers'] = persistent_workers
                loader_kwargs['prefetch_factor'] = prefetch_factor
            self.loaders[dataset_type] = DataLoader(**loader_kwargs)

    def initial(self, dataset_type):
        self.iter_dataset_type = dataset_type
        self._iter_index = 0
        self.iter_loader = iter(self.loaders[dataset_type])

    def _dataset_initial(self, dataset_prefix, params_dict, data_root):
        return data_dataset.dataset(
            params_dict,
            os.path.join(data_root, params_dict['TRAIN']['PATH']['DATASET_PATH'][0]),
            dataset_prefix,
            self.res_dict,
        )

    def get_dataset(self, dataset_prefix, original_data=False):
        return self._dataset_initial(dataset_prefix, self.params_dict, self.data_root)

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_loader is None:
            raise StopIteration

        inputs, label, filenames = next(self.iter_loader)
        if inputs is None or (label is None and self.train_flag):
            raise StopIteration

        if not hasattr(self, '_iter_index') or self.iter_dataset_type is None:
            self._iter_index = 0
        self._iter_index += 1
        return inputs, label, filenames, self._iter_index
