#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 18:39:04 2020

@author: lz
"""
import os
import yaml
import gc
import torch
import sys
import time
sys.path.insert(0, os.path.dirname(__file__))
from ModelUtils.model_components import model_components
from DataUtils.data_components import data_components
from model_predict import Predictor


class LidarDetector(object):
    def __init__(self, params_dict, data_root, result_root, model_epoch_root='', pretrained_path=None, load_model=True, load_data=True, train_flag=True, test_flag=True, check_flag=False, res_dict={}):
        self.params_dict = params_dict
        self.data_root = data_root
        self.result_root = result_root
        self.model_epoch_root = model_epoch_root
        if not os.path.exists(self.result_root):
            if train_flag or test_flag:
                os.makedirs(self.result_root)
        if self.model_epoch_root != '' and not os.path.exists(self.model_epoch_root):
            os.makedirs(self.model_epoch_root)
        self.pretrained_path = pretrained_path
        self.train_flag = train_flag
        self.test_flag = test_flag
        self.check_flag = check_flag
        self.res_dict = res_dict
        if 'msg' not in self.res_dict:
            self.res_dict['msg'] = []
        
        if load_model:
            self.model_component = model_components(self.params_dict.copy(), self.result_root, self.model_epoch_root, self.pretrained_path, res_dict=self.res_dict)
        if load_data:
            self.data_component = data_components(self.params_dict.copy(), self.data_root, self.result_root, self.train_flag, self.test_flag, res_dict=self.res_dict)

    def train(self):
        # self.model_component.model_computer.load_model_params_torch(self.pretrained_path)
        # if not self.params_dict['TRAIN']['OVERALL']['INITIAL_RESULT'][0] and self.params_dict['TRAIN']['CTRL']['CTRL_']['CONTINUE_EPOCH'][0] is not None:
        #     start_epoch = self.params_dict['TRAIN']['CTRL']['CTRL_']['CONTINUE_EPOCH'][0] + 1
        # else:
            # start_epoch = 0
        
        start_epoch = 0

        if self.params_dict['TRAIN']['CTRL']['CTRL_']['SCHEDULER_TYPE'][0] == 'OneCycleLR':
            total_iters = self.params_dict['TRAIN']['CTRL']['CTRL_']['EPOCH_NUM'][0] * self.data_component.data_loader.dataset['train'].num_batch
            last_step = start_epoch * self.data_component.data_loader.dataset['train'].num_batch - 1 if start_epoch != 0 else -1
            if last_step != -1:
                self.model_component.optimizer.param_groups[0]['initial_lr'] = 1e-4
                self.model_component.optimizer.param_groups[0]['max_lr'] = 0.001
                self.model_component.optimizer.param_groups[0]['min_lr'] = 1e-9
                self.model_component.optimizer.param_groups[0]['max_momentum'] = 0.95
                self.model_component.optimizer.param_groups[0]['base_momentum'] = 0.85
            self.model_component.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.model_component.optimizer,
                                                                                 max_lr=0.001, total_steps=total_iters,
                                                                                 pct_start=0.4, last_epoch=last_step,
                                                                                 div_factor=10, final_div_factor=1e4)
                                                                                
                                                                   
        for epo in range(start_epoch, self.params_dict['TRAIN']['CTRL']['CTRL_']['EPOCH_NUM'][0]):
            gc.collect()
            torch.cuda.empty_cache()
            self.res_dict['msg'].append("### %d/%d epoch ###" % (epo, self.params_dict['TRAIN']['CTRL']['CTRL_']['EPOCH_NUM'][0]))
            self.train_one_epoch()
            gc.collect()
            torch.cuda.empty_cache()

            self.model_component.loss_computer.save()
            self.model_component.save_model_torch(epo)
            
            self.test_one_epoch(test_type='eva_vali')
            gc.collect()
            torch.cuda.empty_cache()

            if self.check_flag:
                break
            
        self.model_component.loss_computer.save()
    
    def test(self, epoch_list, save=False):
        for epo in epoch_list:
            gc.collect()
            torch.cuda.empty_cache()
            self.model_component.model_computer.load_model_params_torch(epo)
            self.res_dict['msg'].append("evaluate result of %d epoch" % epo)

            self.test_one_epoch(test_type='eva_test', save=save)
            gc.collect()
            torch.cuda.empty_cache()
    
    def predict(self, predictions_root, epoch_id):
        gc.collect()
        torch.cuda.empty_cache()
        self.res_dict['msg'].append('predict the result of %d epoch' % epoch_id)
        predictor = Predictor(self.get_model(epoch_id), self.get_dataloader(), self.data_root, 
                              predictions_root, self.params_dict, self.check_flag, self.res_dict)
        predictor.Predict()         
        self.res_dict['msg'].append("############### Prediction End ###############")
        self.res_dict['msg'].append("############### Prediction End ###############")
        gc.collect()
        torch.cuda.empty_cache()
    
    def get_model(self, epoch=None):
        if epoch is not None:
            self.model_component.model_computer.load_model_params_torch(epoch)
        return self.model_component.model_computer.model
    
    def get_dataset(self, dataset_prefix, original_data=True):
        return self.data_component.data_loader.get_dataset(dataset_prefix, original_data=original_data)
    
    def get_dataloader(self):
        return self.data_component.data_loader
    
    def save_bin_epoch(self, epoch):
        self.model_component.model_computer.save_model_params_bin(epoch)
        
    def save_onnx_epoch(self, epoch):
        self.model_component.model_computer.save_model_params_onnx(epoch)
    
    def train_one_epoch(self):
        self.model_component.model_computer.model.train()
        
        if self.params_dict['TRAIN']['MODEL']['FREEZE']['LAYER'][0]:
            self.model_component.model_computer.model_freeze(self.params_dict['TRAIN']['MODEL']['FREEZE']['LAYER_NAMES'][0])
    
        self.data_component.data_loader.initial('train')
        
        self.res_dict['msg'].append("lr: %.6f" % (self.model_component.optimizer.param_groups[0]['lr']))
                
        for inputs, labels, filenames, idx in self.data_component.data_loader:
            gc.collect()
            torch.cuda.empty_cache()
            if idx % self.params_dict['TRAIN']['CTRL']['LOSS']['PRINT_GAP'][0] == 0:
                self.res_dict['msg'].append("%d/%d losses: " % (idx, self.data_component.data_loader.dataset_length['train']))
            
            self.model_component.optimizer.zero_grad()
            time.sleep(0.05)
            outputs = self.model_component.model_computer.model_compute(inputs)
            loss = self.model_component.loss_computer.loss_compute(outputs, labels, idx % self.params_dict['TRAIN']['CTRL']['LOSS']['PRINT_GAP'][0] == 0)

            if loss == 0:
                continue
            
            loss.backward()
            
            self.model_component.optimizer.step()
            if isinstance(self.model_component.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.model_component.scheduler.step()
                if idx % self.params_dict['TRAIN']['CTRL']['LOSS']['PRINT_GAP'][0] == 0:
                    self.res_dict['msg'].append("%d/%d lr: %.6f " % (idx, self.data_component.data_loader.dataset_length['train'],
                                                                     self.model_component.optimizer.param_groups[0]['lr']))
                    print("%d/%d lr: %.6f " % (idx, self.data_component.data_loader.dataset_length['train'],
                                                                     self.model_component.optimizer.param_groups[0]['lr']))

            gc.collect()
            torch.cuda.empty_cache()

            if self.check_flag: 
                break

        if not isinstance(self.model_component.scheduler, torch.optim.lr_scheduler.OneCycleLR):
            self.model_component.scheduler.step()
        
    def test_one_epoch(self, test_type='eva_vali', save=True):
        self.model_component.model_computer.model.eval()
        
        print("########## evaluate in %s ##########" % test_type)
        self.res_dict['msg'].append("########## evaluate in %s ##########" % (test_type))
        
        self.data_component.data_loader.initial(test_type)
        
        self.data_component.data_evaluater.initial()
        
        with torch.no_grad():
            for inputs, labels, filenames, idx in self.data_component.data_loader:
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(0.05)
                outputs = self.model_component.model_computer.model_compute(inputs)
                self.data_component.data_evaluater.record(outputs, labels)
                gc.collect()
                torch.cuda.empty_cache()

                if self.check_flag:
                    break
            
        self.data_component.data_evaluater.evaluate()
        if save:
            self.data_component.data_evaluater.save()
        
        print('############### end ###############')
        self.res_dict['msg'].append("############### end ###############")


