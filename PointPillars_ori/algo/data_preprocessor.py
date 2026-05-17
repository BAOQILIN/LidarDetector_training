import os
import random

import numpy as np
import json
import pandas as pd
import yaml
import pickle


class DataPreprocessor(object):
    def __init__(self, params_dict, data_root, train_flag, test_flag, res_dict={}):
        """
            此类实现 原始数据集 --> 预处理完毕的数据集文件(在data_dataset.py中被使用)
            如果是模型训练,要求预处理完毕的数据集文件已经划分 训练集 和 验证集文件,从而在data_dataset.py中可以直接使用
            如果是模型测试/感知,则只划分一个完整的测试集文件
            params_dict: config_algo_template.yaml中 PREPROCESS模块 的所有参数
            data_root: 原始数据集的根目录以及保存预处理完毕的数据集文件根目录,文件层级结构如下:
            data_root
                |---- PointCloud
                |       |---- XXX_YYY_ZZZ0.pcd
                |       |---- XXX_YYY_ZZZ1.pcd
                |       |---- ...(会同时包含对应的左/主/右雷达点云数据)
                |---- Image
                |       |---- AAA_BBB_CCC0.jpg
                |       |---- AAA_BBB_CCC1.jpg
                |       |---- ...(会同时包含对应的前/后/左/右相机图片数据)
                |---- Label
                |       |---- label.json(标签文件模板)
                |       |---- XXX_YYY_ZZZ0.json
                |       |---- XXX_YYY_ZZZ1.json
                |       |---- AAA_BBB_CCC0.json
                |       |---- AAA_BBB_CCC1.json
                |       |---- ...(会同时包含对应的所有传感器的标签数据)
                |---- params_dict["SAVE_DATA_PATH"][0] --> 用户在配置文件 PREPROCESS模块 设定的保存预处理后数据的文件夹名称
                |       |---- file1
                |       |---- file2
                |       |---- ...
            原始数据集的路径：os.path.join(data_root, params_dict["ORI_DATA_PATH"][0])
            预处理完毕的数据集文件保存路径：os.path.join(data_root, params_dict["SAVE_DATA_PATH"][0])
            注意：由于多个同类传感器的原始数据存储在同一个文件夹下,数据预处理时用户可能需要自行根据文件名中的相关字符(eg:main/left/right...)提取出指定传感器的原始数据

            train_flag: True/False,说明当前数据预处理过程是用于后续的 模型训练 or 模型测试/感知.
                        当用于模型训练时,应当基于一定的比例划分出训练集和验证集两个文件;当用于模型测试/感知时,则只用生成一整个文件即可
            test_flag: True/False, 说明当前数据预处理过程是用于模型测试 or 模型感知
                        当用于模型测试,则应该包含标签数据; 当用于模型感知,则只需要包含原始数据即可.
            通过train_flag 和 test_flag 两个标志参数, 区分了训练过程 / 测试过程 / 感知过程.
            res_dict: res_dict['msg']是一个列表,列表中的每个元素均为字符串,用户通过向此列表中添加字符串,在web前端页面打印相应的消息
        """
        self.params_dict = params_dict
        self.data_root = data_root
        self.ori_data_dir = os.path.join(self.data_root, self.params_dict['ORI_DATA_PATH'][0])
        # if train_flag or test_flag:
        self.ori_label_dir = os.path.join(self.data_root, self.params_dict['ORI_LABEL_PATH'][0])
        self.label_template_path = os.path.join(self.ori_label_dir, 'label.json')
        if not os.path.exists(self.label_template_path):
            self.res_dict.append('Error, No label.json!')
        self.rectified_data_dir = os.path.join(self.data_root, self.params_dict['SAVE_DATA_PATH'][0])
        if not os.path.exists(self.rectified_data_dir):
            os.makedirs(self.rectified_data_dir)

        self.train_flag = train_flag
        self.test_flag = test_flag
        self.res_dict = res_dict
        if 'msg' not in self.res_dict:
            self.res_dict['msg'] = []

        self.dataset_list = []

    def data_preprocess(self):
        """
            数据预处理的唯一固定接口，不可删除，数据预处理过程的具体实现
            原始数据集 --> 预处理完毕的数据集文件(在data_dataset.py中被使用)
            为了后续使用方便，当前默认原始数据标签格式为数据管理平台标签格式,同时需要存在label.json文件
            但这导致短期内无法测试
        """
        # 读取label.json文件获取class_name和id的映射关系
        id_2_name_map = {}
        if self.train_flag or self.test_flag:
            with open(self.label_template_path, 'rb') as f:
                temp = json.load(f, encoding='utf-8')
            for each in temp:
                if each['task'] == '目标检测':
                    label_json = each['label']
                    for obj in label_json:
                        id_2_name_map[obj['id']] = obj['name']

        # id_2_name_map = {26: 'Pedestrian', 39: 'Mbike', 157: 'Car', 156: 'Bus', 229: 'Tricycle',
        #                  230: 'Pedestrian', 227: 'Mbike',}

        roi = self.params_dict['ROI']
        point_cloud_range = []
        dims = ['X', 'Y', 'Z']
        for dim in dims:
            point_cloud_range.append(roi[f'{dim}_MIN'][0])
        for dim in dims:
            point_cloud_range.append(roi[f'{dim}_MAX'][0])

        point_file_list = os.listdir(self.ori_data_dir)
        for point_file in point_file_list:
            # 筛选出指定雷达的点云(待提供文件名样式)
            lidar_path = os.path.join(self.ori_data_dir, point_file)
            if not self.train_flag and not self.test_flag:  # 感知预测阶段
                self.dataset_list.append({'lidar_path': lidar_path, 'gt_boxes': np.random.randint(0, 0, size=[0, 7]),
                                          'gt_names': np.array([]), 'num_lidar_pts': np.array([])})
            else:  # 训练or测试阶段都需要真值标签
                label_file_list = os.listdir(self.ori_label_dir)
                name, extension = os.path.splitext(point_file)
                label_file = name + '.json'
                if label_file in label_file_list:  # 存在真值标签文件
                    points = pd.read_csv(lidar_path, skiprows=11, header=None, sep=' ').values.astype(np.float32)
                    
                    mask = self.mask_points_by_range(points, point_cloud_range)
                    points = points[mask]
                    with open(os.path.join(self.ori_label_dir, label_file), 'rb') as f:
                        label_dict = json.load(f)
                        for each in label_dict:
                            if each['task'] == '目标检测':
                                label_object3d = each['annotation']['annotation']
                                num_object = len(label_object3d)
                                if num_object > 0:  # 目标数量不为0
                                    gt_boxes = []
                                    gt_names = []
                                    for i in range(num_object):
                                        object_label = label_object3d[i]
                                        coord = object_label['position']
                                        sizes = object_label['dimension']
                                        rotation = object_label['rotation'][-1:]
                                        gt_boxes.append(coord + sizes + rotation)
                                        gt_names.append(id_2_name_map[object_label['label_id']])
                                    gt_boxes = np.array(gt_boxes, dtype=np.float32).reshape(-1, 7)
                                    gt_names = np.array(gt_names)
                                    num_lidar_pts = self.get_object_pointnum(points, gt_boxes[:, :3], gt_boxes[:, 3:6], gt_boxes[:, -1:])
                                    mask = self.mask_points_by_range(gt_boxes[:, :3], point_cloud_range)
                                    gt_boxes = gt_boxes[mask]
                                    gt_names = gt_names[mask]
                                    num_lidar_pts = num_lidar_pts[mask]
                                    self.dataset_list.append({'lidar_path': lidar_path, 'gt_boxes': gt_boxes,
                                                              'gt_names': gt_names, 'num_lidar_pts': num_lidar_pts})
        num_samples = len(self.dataset_list)
        self.res_dict['msg'].append('Total sample number: %s' % num_samples)

        if self.train_flag:  # 训练阶段
            num_train = int(num_samples * self.params_dict['RATIO'][0])
            random.shuffle(self.dataset_list)
            with open(os.path.join(self.rectified_data_dir, 'training.pkl'), 'wb') as f:
                pickle.dump(self.dataset_list[:num_train], f)
            with open(os.path.join(self.rectified_data_dir, 'validation.pkl'), 'wb') as f:
                pickle.dump(self.dataset_list[num_train:], f)

        else:  # 非训练过程,不需要切分数据集
            if self.test_flag:  # 测试集
                with open(os.path.join(self.rectified_data_dir, 'testing.pkl'), 'wb') as f:
                    pickle.dump(self.dataset_list, f)
            else:  # 感知预测原始数据
                with open(os.path.join(self.rectified_data_dir, 'prediction.pkl'), 'wb') as f:
                    pickle.dump(self.dataset_list, f)


    @staticmethod
    def get_object_pointnum(points, object_center, size, heading):
        """
        根据点云和目标框的属性 计算框中点的数量
        :param points:
        :param object_center:
        :param size:
        :param heading:
        :return:
        """
        nan_index = np.isnan(points)
        points[nan_index] = 9999
        object_num = object_center.shape[0]
        object_pointnum_list = []
        for i in range(object_num):
            points_re = points[:, :3] - object_center[i, :]
            rot_z_matrix = np.identity(3)
            rot_z_matrix[0][0] = np.cos(heading[i, 0])
            rot_z_matrix[1][0] = np.sin(heading[i, 0])
            rot_z_matrix[0][1] = -np.sin(heading[i, 0])
            rot_z_matrix[1][1] = np.cos(heading[i, 0])
            points_re_rot = np.dot(points_re, rot_z_matrix)
            x_min = - (size[i, 0] / 2)
            x_max = size[i, 0] / 2
            y_min = - (size[i, 1] / 2)
            y_max = size[i, 1] / 2
            z_min = - (size[i, 2] / 2)
            z_max = size[i, 2] / 2
            point_index = np.where((x_min <= points_re_rot[:, 0]) & (points_re_rot[:, 0] <= x_max) &
                                   (y_min <= points_re_rot[:, 1]) & (points_re_rot[:, 1] <= y_max) &
                                   (z_min <= points_re_rot[:, 2]) & (points_re_rot[:, 2] <= z_max), True, False)
            object_pointnum_list.append(len(points_re_rot[point_index]))
        return np.array(object_pointnum_list)

    @staticmethod
    def mask_points_by_range(points, limit_range):
        mask = (points[:, 0] > limit_range[0]) & (points[:, 0] <= limit_range[3]) & (
                    points[:, 1] > limit_range[1]) & (points[:, 1] <= limit_range[4])
        return mask

        # ...
        # self.func0(param0, param1, ...)
        # self.func1(param0, param1, param2, ...)
        # ...

    # def func0(self, arg0, arg1, ...):
    #     """示例：自定义方法0，可删"""
    #
    # def func1(self, arg0, arg1, arg2, ...):
    #     """示例：自定义方法1，可删"""
    #
    # ...
    # "自行增加类内自定义方法"

# def function0(arg0, arg1, ...):
#     """示例：自定义类外函数0，可删"""
#
# ...
# "自行增加类外自定义函数"


if __name__ == '__main__':
    cfg_file_path = r'D:\Web_LidarDetector\Pointnet\algo\timestamp\algo_config.yaml'
    params_dict = yaml.load(open(cfg_file_path, encoding='utf-8'), Loader=yaml.FullLoader)
    data_root = r'D:\HBOX_Project\LidarDataset\main_dataset\R80_UrbanRoad_20210731_120612'
    train_flag = True
    res_dict = {'msg': []}
    data_preprocessor = DataPreprocessor(params_dict, data_root, train_flag, res_dict)
    data_preprocessor.data_preprocess()
