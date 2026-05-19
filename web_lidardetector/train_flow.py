#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 18:39:04 2020

@author: lz
"""
import gc
import os
from datetime import datetime
import yaml
import torch
import sys
import time
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'PointPillars', 'algo'))
from ModelUtils.model_components import model_components
from DataUtils.data_components import data_components
from model_predict import Predictor
from utils import points_to_voxel_batch_gpu


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
        continue_epoch = self.params_dict['TRAIN']['CTRL']['CTRL_'].get('CONTINUE_EPOCH', [-1])[0]
        should_resume = (not self.params_dict['TRAIN']['OVERALL']['INITIAL_RESULT'][0]) and continue_epoch is not None and continue_epoch >= 0
        if should_resume:
            start_epoch = continue_epoch + 1
            continue_prefix = self.params_dict['TRAIN']['PATH'].get('CONTINUE_MODEL_PREFIX', [self.params_dict['TRAIN']['PATH']['SAVE_MODEL_PREFIX'][0]])[0]
            self.model_component.model_computer.load_model_params_torch(continue_epoch, prefix=continue_prefix)
            resume_message = self._timestamped(f'Train: resume from checkpoint epoch {continue_epoch}, start epoch {start_epoch + 1}')
        else:
            start_epoch = 0
            resume_message = self._timestamped('Train: start from epoch 1')

        self.res_dict['msg'].append(resume_message)
        print(resume_message)

        if self.params_dict['TRAIN']['CTRL']['CTRL_']['SCHEDULER_TYPE'][0] == 'OneCycleLR':
            train_batches = len(self.data_component.data_loader.loaders['train'])
            total_iters = self.params_dict['TRAIN']['CTRL']['CTRL_']['EPOCH_NUM'][0] * train_batches
            last_step = start_epoch * train_batches - 1 if start_epoch != 0 else -1
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
                                                                                
                                                                   
        eval_every_epochs = self.params_dict['TRAIN']['CTRL']['CTRL_'].get('EVAL_EVERY_EPOCHS', [1])[0]
        for epo in range(start_epoch, self.params_dict['TRAIN']['CTRL']['CTRL_']['EPOCH_NUM'][0]):
            total_epochs = self.params_dict['TRAIN']['CTRL']['CTRL_']['EPOCH_NUM'][0]
            epoch_start_line = self._timestamped(f'Train: epoch {epo + 1}/{total_epochs} start')
            self.res_dict['msg'].append(epoch_start_line)
            print(epoch_start_line)
            self.train_one_epoch(epo, total_epochs)

            self.model_component.loss_computer.save()
            self.model_component.save_model_torch(epo)

            should_eval = ((epo + 1) % max(1, eval_every_epochs) == 0) or self.check_flag or (epo + 1 == total_epochs)
            if should_eval:
                self.test_one_epoch(test_type='eva_vali', epoch_index=epo, total_epochs=total_epochs)

            if self.check_flag:
                break
            
        self.model_component.loss_computer.save()
    
    def test(self, epoch_list, save=False):
        for epo in epoch_list:
            gc.collect()
            torch.cuda.empty_cache()
            self.model_component.model_computer.load_model_params_torch(epo)
            self.res_dict['msg'].append(self._timestamped("evaluate result of %d epoch" % epo))

            self.test_one_epoch(test_type='eva_test', save=save)
            gc.collect()
            torch.cuda.empty_cache()
    
    def predict(self, predictions_root, epoch_id):
        gc.collect()
        torch.cuda.empty_cache()
        self.res_dict['msg'].append(self._timestamped('predict the result of %d epoch' % epoch_id))
        predictor = Predictor(self.get_model(epoch_id), self.get_dataloader(), self.data_root, 
                              predictions_root, self.params_dict, self.check_flag, self.res_dict)
        predictor.Predict()         
        self.res_dict['msg'].append(self._timestamped("############### Prediction End ###############"))
        self.res_dict['msg'].append(self._timestamped("############### Prediction End ###############"))
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

    @staticmethod
    def _move_to_device(data):
        if isinstance(data, list):
            return [LidarDetector._move_to_device(item) for item in data]
        if isinstance(data, tuple):
            return tuple(LidarDetector._move_to_device(item) for item in data)
        if torch.is_tensor(data) and torch.cuda.is_available():
            return data.cuda(non_blocking=True)
        return data

    @staticmethod
    def _timestamped(message):
        timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S:%f')[:-3]
        return f'[{timestamp}] {message}'

    def train_one_epoch(self, epoch_index, total_epochs):
        self.model_component.model_computer.model.train()

        if self.params_dict['TRAIN']['MODEL']['FREEZE']['LAYER'][0]:
            self.model_component.model_computer.model_freeze(self.params_dict['TRAIN']['MODEL']['FREEZE']['LAYER_NAMES'][0])

        self.data_component.data_loader.initial('train')
        total_iters = len(self.data_component.data_loader.loaders['train'])
        print_gap = self.params_dict['TRAIN']['CTRL']['LOSS']['PRINT_GAP'][0]
        data_wait_start = time.time()

        for inputs, labels, filenames, idx in self.data_component.data_loader:
            data_ready_time = time.time()
            # GPU voxelization (inputs is points_list from collate_batch)
            vp = self.data_component.data_loader.get_voxel_params()
            voxels, coors, num_pts = points_to_voxel_batch_gpu(
                inputs, vp['voxel_size'], vp['point_cloud_range'],
                vp['max_num_points'], vp['max_voxels'])
            inputs = [voxels, num_pts.float(), coors.float()]
            labels = self._move_to_device(labels)
            self.model_component.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = self.model_component.model_computer.model_compute(inputs)
                loss = self.model_component.loss_computer.loss_compute(outputs, labels, idx % print_gap == 0)

            if loss == 0:
                data_wait_start = time.time()
                continue

            self.model_component.grad_scaler.scale(loss).backward()
            self.model_component.grad_scaler.step(self.model_component.optimizer)
            self.model_component.grad_scaler.update()
            if isinstance(self.model_component.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.model_component.scheduler.step()

            if idx % print_gap == 0:
                metrics = self.model_component.loss_computer.latest_metrics
                batch_end_time = time.time()
                data_time = data_ready_time - data_wait_start
                batch_time = batch_end_time - data_ready_time
                log_line = (
                    f'Train: epoch {epoch_index + 1}/{total_epochs}, '
                    f'iter {idx}/{total_iters}, '
                    f'lr {self.model_component.optimizer.param_groups[0]["lr"]:.6f}, '
                    f'loss {metrics.get("total_loss", 0.0):.4f}, '
                    f'cls {metrics.get("cls_loss", 0.0):.4f}, '
                    f'reg {metrics.get("reg_loss", 0.0):.4f}, '
                    f'data_time {data_time:.3f}s, '
                    f'batch_time {batch_time:.3f}s'
                )
                timestamped_line = self._timestamped(log_line)
                self.res_dict['msg'].append(timestamped_line)
                print(timestamped_line)

            data_wait_start = time.time()

            if self.check_flag:
                break

        if not isinstance(self.model_component.scheduler, torch.optim.lr_scheduler.OneCycleLR):
            self.model_component.scheduler.step()
        
    def test_one_epoch(self, test_type='eva_vali', save=True, epoch_index=None, total_epochs=None):
        self.model_component.model_computer.model.eval()

        eval_prefix = f'Eval: epoch {epoch_index + 1}/{total_epochs}, ' if epoch_index is not None and total_epochs is not None else 'Eval: '
        eval_start = time.time()
        eval_banner = self._timestamped(f'{eval_prefix}stage {test_type} start')
        print(eval_banner)
        self.res_dict['msg'].append(eval_banner)

        self.data_component.data_loader.initial(test_type)
        self.data_component.data_evaluater.initial()

        total_batches = len(self.data_component.data_loader.loaders[test_type])
        print_gap = max(1, total_batches // 20)  # ~5% progress steps

        with torch.no_grad():
            for inputs, labels, filenames, idx in self.data_component.data_loader:
                vp = self.data_component.data_loader.get_voxel_params()
                voxels, coors, num_pts = points_to_voxel_batch_gpu(
                    inputs, vp['voxel_size'], vp['point_cloud_range'],
                    vp['max_num_points'], vp['max_voxels'])
                inputs = [voxels, num_pts.float(), coors.float()]
                labels = self._move_to_device(labels)
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    outputs = self.model_component.model_computer.model_compute(inputs)
                self.data_component.data_evaluater.record(outputs, labels)
                if idx % print_gap == 0:
                    eta = (time.time() - eval_start) / max(idx, 1) * (total_batches - idx)
                    progress_msg = self._timestamped(f'{eval_prefix}stage {test_type} batch {idx}/{total_batches}, eta {eta:.0f}s')
                    print(progress_msg)
                    self.res_dict['msg'].append(progress_msg)
                if self.check_flag:
                    break

        self.data_component.data_evaluater.evaluate()
        if save:
            self.data_component.data_evaluater.save()

        eval_summary = self._timestamped(f'{eval_prefix}stage {test_type} done, elapsed {time.time() - eval_start:.3f}s')
        print(eval_summary)
        self.res_dict['msg'].append(eval_summary)


