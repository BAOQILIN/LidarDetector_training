# from Web_LidarDetector.train_flow import LidarDetector
# from Pointnet.algo.timestamp.data_displayer import DataDisplayer
# from Pointnet.algo.timestamp.data_preprocessor import DataPreprocessor
# import yaml
# import torch

import yaml
import os
import sys
import queue
Base_dir = os.path.dirname(__file__)
sys.path.insert(0, Base_dir)
temp1 = os.path.join(Base_dir, 'PointPillars/algo')
temp2 = os.path.join(Base_dir, 'PointPillars/model/model')
temp3 = os.path.join(Base_dir, 'PointPillars/model/layer')
sys.path.append(temp1)
sys.path.append(temp2)
sys.path.append(temp3)

from web_lidardetector.interface import *


if __name__ == "__main__":
    # train and val
    list = []
    for item in sys.path:
        if item == temp1 or item == temp2 or item == temp3:
            list.append(item)

    for item in list:
        sys.path.remove(item)

    temp1 = os.path.join(Base_dir, 'PointPillars/algo')
    temp2 = os.path.join(Base_dir, 'PointPillars/model/model')
    temp3 = os.path.join(Base_dir, 'PointPillars/model/layer')
    sys.path.insert(0, temp1)
    sys.path.insert(0, temp2)
    sys.path.insert(0, temp3)
    reload_list = [temp1, temp2, temp3]

    config_file = r'D:\Web_LidarDetector_test\PointPillars\algo\algo_config.yaml'
    params_dict = yaml.load(open(config_file, encoding='utf-8'), Loader=yaml.FullLoader)
    data_root = r'D:\Web_LidarDetector_test\home\data_root'
    result_root = r'D:\Web_LidarDetector_test\home\result_root'
    model_epoch_root = r'D:\Web_LidarDetector_test\home\result_root\model_epoch'
    picture_root = r'D:\Web_LidarDetector_test\home\result_root\pictures'
    predictions_root = r'D:\Web_LidarDetector_test\home\result_root\predictions'
    pretrained_path = r'D:\Web_LidarDetector_test\home\result_root\model_epoch\PointPillars_99.torch'
    epoch_list = [0, 1]
    q = queue.Queue()

    ICheck(q, params_dict, data_root, result_root, picture_root, model_epoch_root, pretrained_path, predictions_root, reload_list, check_flag=True, res_dict={})

    IPreprocess_ITrain(q, params_dict, data_root, result_root, picture_root, model_epoch_root, reload_list, pretrained_path, res_dict={})
    
    IPreprocess_ITest(q, params_dict, data_root, result_root, model_epoch_root, reload_list, epoch_list, res_dict={})
    
    ISave(q, params_dict, data_root, result_root, model_epoch_root, reload_list, save_epoch=4, check_flag=False, res_dict={})

    IPreprocess_Predict(q, params_dict, data_root, "", model_epoch_root, predictions_root, reload_list, epoch_id=4, res_dict={})

    print('Done!')

