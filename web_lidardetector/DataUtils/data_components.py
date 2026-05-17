#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 16:49:04 2020

@author: lz
"""

from importlib import reload
from DataUtils.data_loader import data_loader
import data_evaluater


class data_components(object):
    def __init__(self, params_dict, data_root, result_root, train_flag, test_flag, res_dict={}):
        self.params_dict = params_dict
        self.data_root = data_root
        self.result_root = result_root
        self.train_flag = train_flag
        self.test_flag = test_flag
        self.res_dict = res_dict
        if 'msg' not in self.res_dict:
            self.res_dict['msg'] = []
        
        self.__initial_data_loader()
        self.__initial_data_evaluater()
    
    def __initial_data_loader(self):
        self.data_loader = data_loader(self.params_dict.copy(), self.data_root, self.train_flag, self.test_flag, res_dict=self.res_dict)
            
    def __initial_data_evaluater(self):
        self.data_evaluater = data_evaluater.data_evaluater(self.params_dict.copy(), self.result_root, self.train_flag, res_dict=self.res_dict)

    def __set_params(self, params_dict):
        for k, v in params_dict.items():
            exec('self.' + k + '=v')