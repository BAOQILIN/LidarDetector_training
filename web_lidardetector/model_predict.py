'''
Description: 
Version: 
Author: yining.jin
Date: 2023-01-05 10:28:43
LastEditors: yining.jin
LastEditTime: 2023-03-14 13:56:15
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 10:43:42 2020

@author: lz
"""

import os
import json
import torch
import time
import data_postprocessor


class Predictor(object):
    def __init__(self, model, data_loader, data_root, predictions_root, params_dict, check_flag=False, res_dict={}):
        self.model = model
        self.data_loader = data_loader
        self.data_root = data_root
        self.predictions_root = predictions_root
        if not os.path.exists(self.predictions_root):
            os.makedirs(self.predictions_root)
        self.params_dict = params_dict
        self.check_flag = check_flag
        self.res_dict = res_dict
        if 'msg' not in self.res_dict:
            self.res_dict['msg'] = []

        self.label_template_path = os.path.join(self.data_root, 'Label', 'label.json')
        if not os.path.exists(self.label_template_path):
            self.res_dict['msg'].append('Predicting Error, No label.json!')

        self.data_postprocessor = data_postprocessor.data_postprocessor(self.label_template_path, self.params_dict)

    def Predict(self):
        if not os.path.exists(self.predictions_root):
            os.makedirs(self.predictions_root)
        self.model.eval()
        self.data_loader.initial('eva_predict')
        with torch.no_grad():
            print('Starting predicting data, it may take several minutes...')
            self.res_dict['msg'].append('Starting predicting data, it may take several minutes...')
            for inputs, labels, filenames, idx in self.data_loader:  ## for each batch
                time.sleep(0.05)
                outputs = self.model(inputs)
                predictions, filenames = self.data_postprocessor.data_postprocess(outputs, filenames)

                for prediction, filename in zip(predictions, filenames):
                    filename = os.path.split(filename)[-1]
                    name, extension = os.path.splitext(filename)
                    label_path = os.path.join(self.data_root, 'Label', name + '.json')
                    if os.path.exists(label_path):
                        with open(label_path, 'rb') as f:
                            label_list = json.load(f)
                        for label in label_list:
                            for task_annotation in prediction:
                                if label['task'] == task_annotation['task']:
                                    label['annotation']['annotation'] = task_annotation['annotation']['annotation']
                    else:
                        label_list = prediction

                    save_path = os.path.join(self.predictions_root, name + '.json')
                    with open(save_path, 'w', encoding='utf-8') as f:
                        json.dump(label_list, f, ensure_ascii=False, indent=4)
                
                self.res_dict['msg'].append("Predicting batch %d" % idx)

                if self.check_flag:
                    break
            print('Prediction Done!')
            self.res_dict['msg'].append('Prediction Done!')


    
    