'''
Description: 
Version: 
Author: yining.jin
Date: 2023-03-01 16:24:54
LastEditors: yining.jin
LastEditTime: 2023-03-13 18:38:06
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 19:08:51 2020

@author: lz
"""
from importlib import reload
import os
import torch.optim as optim
import model_computers
import loss_computers
from ModelUtils.utils import get_cosine_schedule_with_warmup


class model_components(object):
    def __init__(self, params_dict, result_root, model_epoch_root, pretrained_path=None, res_dict={}):
        self.params_dict = params_dict
        self.result_root = result_root
        self.model_epoch_root = model_epoch_root
        self.pretrained_path = pretrained_path
        self.res_dict = res_dict
        if 'msg' not in self.res_dict:
            self.res_dict['msg'] = []
        
        result_path = os.path.join(self.result_root, self.params_dict['TRAIN']['PATH']['RESULT_PATH'][0])
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        self.__initial_loss()
        self.__initial_model()
        self.__initial_model_controller()
    
    def save_model_torch(self, epoch):
        self.model_computer.save_model_params_torch(epoch)
    
    def save_model_bin(self, epoch):
        self.model_computer.save_model_params_bin(epoch)
    
    def save_loss(self):
        self.loss_computer.save()
    
    def __initial_model_controller(self):
        self.optimizer = optim.Adam(self.model_computer.model.parameters(), lr=self.params_dict['TRAIN']['CTRL']['CTRL_']['OPTIMIZER_LR'][0], betas=(0.9, 0.999))
        if self.params_dict['TRAIN']['CTRL']['CTRL_']['SCHEDULER_TYPE'][0] == 'warmup':
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, self.params_dict['TRAIN']['CTRL']['CTRL_']['SCHEDULER_WARMUP_STEPS'][0], self.params_dict['TRAIN']['CTRL']['CTRL_']['EPOCH_NUM'][0])
        elif self.params_dict['TRAIN']['CTRL']['CTRL_']['SCHEDULER_TYPE'][0] == 'OneCycleLR':
            self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.001, total_steps=10000, pct_start=0.4)
        else:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.params_dict['TRAIN']['CTRL']['CTRL_']['SCHEDULER_STEP'][0], gamma=self.params_dict['TRAIN']['CTRL']['CTRL_']['SCHEDULER_GAMMA'][0])
    
    def __initial_loss(self):
        self.loss_computer = loss_computers.loss_computer(self.params_dict.copy(), self.result_root, res_dict=self.res_dict)

    def __initial_model(self):
        self.model_computer = model_computers.model_computer(self.params_dict.copy(), self.model_epoch_root, self.pretrained_path, res_dict=self.res_dict)
        
