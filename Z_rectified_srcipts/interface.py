# -*- coding:utf-8 -*-

from importlib import reload
import yaml
import sys
import os
import threading
import traceback
sys.path.insert(0, os.path.dirname(__file__) + os.sep + '..')

from web_lidardetector.train_flow import *
import data_preprocessor
import data_postprocessor
import data_displayer
import data_dataset
import data_evaluater
import loss_computers
import model_computers
import utils
import networks

def module_reload(reload_list, res_dict):
    if 'msg' not in res_dict:
        res_dict['msg'] = []

    for folder in reload_list:
        file_list = os.listdir(folder)
        if 'utils.py' in file_list:
            reload(utils)
            file_list.remove('utils.py')
        for file_item in file_list:
            file_path, fullname = os.path.split(file_item)
            name, ext = os.path.splitext(fullname)
            if ext == '.py':
                try:
                    exec('reload(' + name + ')')
                except:
                    exc_type, exc_value, exc_trackback = sys.exc_info()
                    if exc_type is not NameError:
                        res_dict['msg'].append(str(repr(traceback.format_exception(exc_type, exc_value, exc_trackback))))
                    else:
                        exec('import ' + name)


def IPreprocess_ITrain(q, params_dict, data_root, result_root, picture_root, model_epoch_root, reload_list, res_dict={}):
    module_reload(reload_list, res_dict)

    if 'msg' not in res_dict:
        res_dict['msg'] = []

    try:
        IPreprocess(params_dict['PREPROCESS'], data_root, train_flag=True, test_flag=False, res_dict=res_dict)
        ITrain(params_dict['TRAIN_MODEL'], data_root, result_root, model_epoch_root, res_dict=res_dict)
        IDisplay(params_dict['DISPLAY'], result_root, picture_root, res_dict=res_dict)

    except:
        q.put(sys.exc_info())


def IPreprocess_ITest(q, params_dict, data_root, result_root, model_epoch_root, reload_list, epoch_list, res_dict={}):
    module_reload(reload_list, res_dict)

    if 'msg' not in res_dict:
        res_dict['msg'] = []

    try:
        IPreprocess(params_dict['PREPROCESS'], data_root, train_flag=False, test_flag=True, res_dict=res_dict)
        ITest(params_dict['TRAIN_MODEL'], data_root, result_root, model_epoch_root, epoch_list, res_dict=res_dict)
    except:
        q.put(sys.exc_info())


def IPreprocess_Predict(q, params_dict, data_root, result_root, model_epoch_root, predictions_root, reload_list, epoch_id, res_dict={}):
    module_reload(reload_list, res_dict)

    if 'msg' not in res_dict:
        res_dict['msg'] = []

    # try:
    # IPreprocess(params_dict['PREPROCESS'], data_root, train_flag=False, test_flag=False, res_dict=res_dict)
    IPredict(params_dict['TRAIN_MODEL'], data_root, result_root, model_epoch_root, predictions_root, epoch_id, res_dict=res_dict)
    # except:
    #     q.put(sys.exc_info())


def IPreprocess(params_dict, data_root, train_flag, test_flag, res_dict={}):
    if 'PREPROCESS' in params_dict.keys():
        params_dict = params_dict['PREPROCESS']

    if 'msg' not in res_dict:
        res_dict['msg'] = []

    data_preprocessor_ = data_preprocessor.DataPreprocessor(params_dict, data_root, train_flag, test_flag, res_dict)
    data_preprocessor_.data_preprocess()


def ITrain(params_dict, data_root, result_root, model_epoch_root, res_dict={}):
    if 'TRAIN_MODEL' in params_dict.keys():
        params_dict = params_dict['TRAIN_MODEL']

    if 'msg' not in res_dict:
        res_dict['msg'] = []
    lidar_detector = LidarDetector(params_dict, data_root, result_root, model_epoch_root, train_flag=True, test_flag=False, res_dict=res_dict)
    lidar_detector.train()
    torch.cuda.empty_cache()


def ITest(params_dict, data_root, result_root, model_epoch_root, epoch_list, res_dict={}):
    if 'TRAIN_MODEL' in params_dict.keys():
        params_dict = params_dict['TRAIN_MODEL']

    if 'msg' not in res_dict:
        res_dict['msg'] = []

    lidar_detector = LidarDetector(params_dict, data_root, result_root, model_epoch_root, train_flag=False, test_flag=True, res_dict=res_dict)

    lidar_detector.test(epoch_list, True)


def ISave(q, params_dict, data_root, result_root, model_epoch_root, reload_list, save_epoch=-1, res_dict={}):
    module_reload(reload_list, res_dict)

    if 'TRAIN_MODEL' in params_dict.keys():
        params_dict = params_dict['TRAIN_MODEL']

    if 'msg' not in res_dict:
        res_dict['msg'] = []

    try:
        if save_epoch == -1:
            save_epoch = params_dict["SAVE"]["SAVE_EPOCH"][0]

        lidar_detector = LidarDetector(params_dict, data_root, result_root, model_epoch_root, train_flag=False, test_flag=False, res_dict=res_dict)

        lidar_detector.save_onnx_epoch(save_epoch)
    except:
        q.put(sys.exc_info())


def IDisplay(params_dict, result_root, picture_root, res_dict={}):

    if 'DISPLAY' in params_dict.keys():
        params_dict = params_dict['DISPLAY']

    if not os.path.exists(picture_root):
        os.makedirs(picture_root)

    if 'msg' not in res_dict:
        res_dict['msg'] = []

    data_displayer_ = data_displayer.DataDisplayer(params_dict, result_root, picture_root, res_dict)
    data_displayer_.display_loss()
    data_displayer_.display_eva()


def IPredict(params_dict, data_root, result_root, model_epoch_root, predictions_root, epoch_id, res_dict={}):
    if 'TRAIN_MODEL' in params_dict.keys():
        params_dict = params_dict['TRAIN_MODEL']

    if 'msg' not in res_dict:
        res_dict['msg'] = []

    lidar_detector = LidarDetector(params_dict, data_root, result_root, model_epoch_root, train_flag=False, test_flag=False, res_dict=res_dict)
    lidar_detector.predict(predictions_root, epoch_id)


if __name__ == '__main__':
    print(111)