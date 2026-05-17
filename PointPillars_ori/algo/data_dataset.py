import os
import math
import numpy as np
import pickle
import pandas as pd
import torch
from torch.autograd import Variable
from torch.utils import data
from utils import *


class dataset(data.Dataset):
    """
    此类的设计旨在实现一个数据集的基本功能,其被训练框架调用并实例化,一个训练过程会生成两个实例化对象,分别对应于训练集和验证集;测试/感知过程则生成一个实例化对象
    关于训练集、验证集、测试集的区分,主要依赖构造函数中的prefix形参,用户可由该参数进行判断是加载训练集文件 or 验证集文件 or 测试集文件 进而生成相应对象.
    """

    def __init__(self, params_dict, folder, prefix, res_dict):
        """
        构造方法，不可删改,函数体可自定义增加内容
        params_dict: config_train_template.yaml文件 TRAIN_MODEL模块 对应的参数字典
        folder: 预处理之后的数据集文件的保存路径 = os.path.join(data_root, params_dict['TRAIN']['PATH']['DATASET_PATH'][0])
        prefix: params_dict['TRAIN']['OVERALL']['TRAIN_PREFIX'][0] 或者 params_dict['VALIDATION']['VALI_']['PREFIX'][0]
                或者params_dict['TEST']['TEST_']['PREFIX'][0] 或者 params_dict['PREDICT']['PREDICT_']['PREFIX'][0]
                根据前缀来明确当前需要生成的是 训练阶段 或者 验证阶段 或者 测试阶段 或者 感知阶段 的数据集,不同阶段应该读取不同的预处理后(data_preprocess.py)的文件.
        res_dict: res_dict['msg']是一个列表,列表中的每个元素均为字符串,用户通过向此列表中添加字符串,在web前端页面打印相应的消息
        """
        super(dataset, self).__init__()
        self.params_dict = params_dict
        self.folder = folder
        self.prefix = prefix
        self.res_dict = res_dict

        self.class_names = ['Pedestrian', 'Mbike', 'Car', 'Bus', 'Tricycle']
        self.batch_size = params_dict['TRAIN']['CTRL']['DATA']['BATCH_SIZE'][0]

        if prefix == self.params_dict['TRAIN']['OVERALL']['TRAIN_PREFIX'][0]:  # 模型训练阶段
            self.dataset_cfg_path = os.path.join(folder, 'training.pkl')
        elif prefix == self.params_dict['VALIDATION']['VALI_']['PREFIX'][0]:  # 模型验证阶段
            self.dataset_cfg_path = os.path.join(folder, 'validation.pkl')
        elif prefix == self.params_dict['TEST']['TEST_']['PREFIX'][0]:  # 模型测试阶段
            self.dataset_cfg_path = os.path.join(folder, 'testing.pkl')
        else:  # 模型感知推理阶段
            self.dataset_cfg_path = os.path.join(folder, 'prediction.pkl')

        self.infos = []
        self.include_data(self.dataset_cfg_path)

        if self.prefix == self.params_dict['TRAIN']['OVERALL']['TRAIN_PREFIX'][0] and \
                self.params_dict['TRAIN']['CTRL']['DATA']['RESAMPLING'][0]:
            self.infos = self.balanced_infos_resampling(self.infos)

        self.total_index = np.arange(len(self.infos))
        self.num_batch = math.ceil(len(self.infos) / self.batch_size)

        dims = ['X', 'Y', 'Z']
        roi = self.params_dict['TRAIN']['CTRL']['DATA']['ROI']
        point_cloud_range = []
        for dim in dims:
            point_cloud_range.append(roi[f'{dim}_MIN'][0])
        for dim in dims:
            point_cloud_range.append(roi[f'{dim}_MAX'][0])
        self.point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        voxel_dict = self.params_dict['TRAIN']['CTRL']['DATA']['VOXEL_SIZE']
        voxel_size = []
        for dim in dims:
            voxel_size.append(voxel_dict[f'{dim}'][0])
        self.voxel_size = np.array(voxel_size, dtype=np.float32)

        if self.prefix == self.params_dict['TRAIN']['OVERALL']['TRAIN_PREFIX'][0] \
                and self.params_dict['TRAIN']['CTRL']['AUGMENT']['EXTRACT_MINOR_TYPE'][0] \
                and self.params_dict['TRAIN']['CTRL']['AUGMENT']['APPEND_TYPES'][0] != 'None' \
                and self.params_dict['TRAIN']['CTRL']['AUGMENT']['APPEND_NUM'][0] != 'None' \
                and self.params_dict['TRAIN']['CTRL']['AUGMENT']['APPEND_PROBS'][0] != 'None':
            self.append_cluster_points = {}
            self.append_cluster_index = {}
            self.append_cluster_labels = {}
            for cls_name in self.params_dict['TRAIN']['CTRL']['AUGMENT']['APPEND_TYPES'][0]:
                augment_file_path = os.path.join(folder, 'data_augment_%s.pkl' % cls_name)
                with open(augment_file_path, 'rb') as f:
                    cluster_points, cluster_index, cluster_labels = pickle.load(f)
                    self.append_cluster_points[cls_name] = cluster_points.astype(np.float32)
                    self.append_cluster_index[cls_name] = cluster_index
                    self.append_cluster_labels[cls_name] = cluster_labels.astype(np.float32)
        
        if self.prefix == self.params_dict['TRAIN']['OVERALL']['TRAIN_PREFIX'][0]\
                and self.params_dict['TRAIN']['CTRL']['AUGMENT']['FILTER_MIN_POINTS'][0]:

            min_points_dict = self.param_dict['TRAIN']['CTRL']['AUGMENT']['MIN_POINTS_NUM'][0]
            self.min_points_each_cls = []
            for cls in self.class_names:
                self.min_points_each_cls.append(min_points_dict[cls.upper()][0])

    def __len__(self):
        """函数头不可删改,返回数据集中样本的个数,必须完善返回值"""
        return len(self.infos)

    def __getitem__(self, index):
        if index >= self.num_batch:
            return None, None, None

        elif index == self.num_batch - 1:  # 最后一个batch
            frame_ids = self.total_index.tolist()
            self.batch_size = len(frame_ids)
            self.total_index = np.arange(len(self.infos))  # 在下一个epoch之前将self.total_index重新赋初值

        else:
            temp_idx = np.random.choice(len(self.total_index), self.batch_size, replace=False)
            frame_ids = self.total_index[temp_idx].tolist()
            self.total_index = np.delete(self.total_index, temp_idx)

        gt_boxes_list = []
        voxels_list = []
        coors_list = []
        num_pts_voxel_list = []
        filename_list = []

        for frame_id in frame_ids:
            sample = self.infos[frame_id]
            lidar_points_path = sample['lidar_path']
            gt_boxes = sample['gt_boxes']
            gt_boxes = gt_boxes.astype(np.float32)
            gt_names = sample['gt_names']
            num_lidar_pts = sample['num_lidar_pts']
            # scene_name = sample['scene']  # 新属性,结合extract_minor_cls一起使用

            points = pd.read_csv(lidar_points_path, skiprows=11, header=None, sep=' ').values.astype(np.float32)
            # lidar_points_path = os.path.join(r'D:/HBOX_Project/pointpillars_training/training_dataset/rectify_dataset', lidar_points_path)
            # points = np.fromfile(lidar_points_path, dtype=np.float32).reshape(-1, 4)

            if self.prefix == self.params_dict['TRAIN']['OVERALL']['TRAIN_PREFIX'][0]:
                if self.params_dict['TRAIN']['CTRL']['AUGMENT']['FILTER_MIN_POINTS'][0]:            
                    lidar_pts_each_cls = [num_lidar_pts[gt_names == cls] for cls in self.class_names]
                    gt_names_each_cls = [gt_names[gt_names == cls] for cls in self.class_names]
                    gt_boxes_each_cls = [gt_boxes[gt_names == cls] for cls in self.class_names]
                    mask = [lidar_pts_each_cls[i] > self.min_points_each_cls[i] - 1 for i in range(len(self.class_names))]

                    gt_names = [gt_names_each_cls[i][mask[i]] for i in range(len(self.class_names))]
                    gt_boxes = [gt_boxes_each_cls[i][mask[i]] for i in range(len(self.class_names))]
                    gt_names = np.concatenate(gt_names)
                    gt_boxes = np.concatenate(gt_boxes, axis=0)

                if self.params_dict['TRAIN']['CTRL']['AUGMENT']['EXTRACT_MINOR_TYPE'][0]:
                    points, gt_boxes, gt_names = self.samples_append(points, gt_boxes, gt_names,
                                                                     self.params_dict['TRAIN']['CTRL']['AUGMENT']['APPEND_TYPES'][0],
                                                                     self.params_dict['TRAIN']['CTRL']['AUGMENT']['APPEND_NUM'][0],
                                                                     self.params_dict['TRAIN']['CTRL']['AUGMENT']['APPEND_PROBS'][0])

                if self.params_dict['TRAIN']['CTRL']['AUGMENT']['RANDOM_FLIP'][0]:
                    gt_boxes, points = random_flip_along_x(gt_boxes, points)
                    gt_boxes, points = random_flip_along_y(gt_boxes, points)

                if 'RANDOM_ROTATION' in self.params_dict['TRAIN']['CTRL']['AUGMENT'].keys():
                    rot_dict = self.params_dict['TRAIN']['CTRL']['AUGMENT']['RANDOM_ROTATION']
                    rot_range = [rot_dict['MIN'][0], rot_dict['MAX'][0]]
                    gt_boxes, points = global_rotation(gt_boxes, points, rot_range)

                if 'RANDOM_SCALINGS' in self.params_dict['TRAIN']['CTRL']['AUGMENT'].keys():
                    scaling_dict = self.params_dict['TRAIN']['CTRL']['AUGMENT']['RANDOM_SCALINGS']
                    scaling_ratio = [scaling_dict['MIN'][0], scaling_dict['MAX'][0]]
                    gt_boxes, points = global_scaling(gt_boxes, points, scaling_ratio)

                gt_boxes[:, 6] = limit_heading(gt_boxes[:, 6], 0.5, 2 * np.pi)

                gt_boxes_mask = np.array([n in self.class_names for n in gt_names], dtype=np.bool_)
                gt_boxes = gt_boxes[gt_boxes_mask]
                gt_names = gt_names[gt_boxes_mask]

            gt_classes = np.array([self.class_names.index(n) + 1 for n in gt_names], dtype=np.int32)
            gt_boxes = np.concatenate((gt_boxes, gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            gt_boxes[np.isnan(gt_boxes)] = 0

            nan_mask = None
            for i in range(points.shape[1]):
                if nan_mask is None:
                    nan_mask = ~np.isnan(points[:, i])
                else:
                    nan_mask *= ~np.isnan(points[:, i])
            points = points[nan_mask]

            max_num_points = self.params_dict['TRAIN']['CTRL']['DATA']['MAX_POINTS_PER_VOXEL'][0]
            max_voxels = self.params_dict['TRAIN']['CTRL']['DATA']['MAX_VOXEL_NUM'][0]

            voxels, coors, num_points_per_voxel = points_to_voxel(points, self.voxel_size,
                                                                        self.point_cloud_range,
                                                                        max_num_points, True, max_voxels)

            gt_boxes_list.append(gt_boxes)
            voxels_list.append(voxels)
            coors_list.append(coors)
            num_pts_voxel_list.append(num_points_per_voxel)
            filename_list.append(lidar_points_path.split(os.sep)[-1])

        max_gt = max([gt_box.shape[0] for gt_box in gt_boxes_list])
        batch_gt_boxes = np.zeros((self.batch_size, max_gt, gt_boxes_list[0].shape[-1]), dtype=np.float32)
        for i in range(self.batch_size):
            batch_gt_boxes[i, :gt_boxes_list[i].shape[0], :] = gt_boxes_list[i]

        batch_voxels = np.concatenate(voxels_list, axis=0)

        temps = []
        for i in range(self.batch_size):
            temp = np.pad(coors_list[i], ((0, 0), (1, 0)), mode='constant', constant_values=i)
            temps.append(temp)
        batch_voxel_coors = np.concatenate(temps, axis=0)

        batch_num_pts_voxel = np.concatenate(num_pts_voxel_list, axis=0)

        if index == self.num_batch - 1:  # 最后一个batch处理完毕后重新设置self.batch_size,否则影响下一个epoch的数据加载
            self.batch_size = self.params_dict['TRAIN']['CTRL']['DATA']['BATCH_SIZE'][0]

        if torch.cuda.is_available():
            batch_voxels = torch.from_numpy(batch_voxels).float().cuda()
            batch_num_pts_voxel = torch.from_numpy(batch_num_pts_voxel).float().cuda()
            batch_voxel_coors = torch.from_numpy(batch_voxel_coors).float().cuda()
            batch_gt_boxes = torch.from_numpy(batch_gt_boxes).float().cuda()
            frame_ids = torch.tensor(frame_ids).float().cuda()


        return [batch_voxels, batch_num_pts_voxel, batch_voxel_coors], \
               [batch_gt_boxes, frame_ids], filename_list

    def include_data(self, dataset_path):
        if not os.path.exists(dataset_path):
            print(dataset_path, 'does not exists!!!')
        else:
            with open(dataset_path, 'rb') as f:
                self.infos.extend(pickle.load(f))
                print('Total samples for %s dataset before resampling: %d' % (self.prefix, len(self.infos)))

    def balanced_infos_resampling(self, infos):
        if self.class_names is None:
            return infos

        cls_infos = {name: [] for name in self.class_names}
        for info in infos:
            # info['gt_names'] = info['gt_names'].reshape(-1).tolist()

            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos[name].append(info)
        # 统计数据集内不同类别目标的比例
        num_object_list = [0] * 5
        for info in infos:
            if info['gt_names'].size == 0:
                continue
            if len(info['gt_names'].shape) == 0:
                info['gt_names'] = np.array([info['gt_names']])
            num_object = info['gt_names'].shape[0]
            gt_names = info['gt_names']
            pt_nums = info['num_lidar_pts']
            for i in range(num_object):
                if gt_names[i] == 'Car' and pt_nums[i] >= 40:
                    num_object_list[0] += 1
                elif gt_names[i] == 'Bus' and pt_nums[i] >= 50:
                    num_object_list[1] += 1
                elif gt_names[i] == 'Tricycle' and pt_nums[i] >= 40:
                    num_object_list[2] += 1
                elif gt_names[i] == 'Mbike' and pt_nums[i] >= 35:
                    num_object_list[3] += 1
                elif gt_names[i] == 'Pedestrian' and pt_nums[i] >= 30:
                    num_object_list[4] += 1
                else:
                    continue
        print('Object distribution before resampling: ',
              [{k: v} for k, v in zip(cls_infos.keys(), num_object_list)])

        original_cls_num = {k: len(v) for k, v in cls_infos.items()}
        print('original_cls_samples: ', original_cls_num)

        duplicated_samples = sum([len(v) for k, v in cls_infos.items()])
        print('Total dupliated samples:', duplicated_samples)

        cls_dist = {k: len(v) / duplicated_samples for k, v in cls_infos.items()}
        print('original cls_samples distribution: ', cls_dist)

        # 指的是理想分布比例与现有分布比例之间的比值关系
        frac = 1.0 / len(self.class_names)
        ratios = [frac / v if v != 0 else 100 for v in cls_dist.values()]
        print('ratios: ', [{k: v} for k, v in zip(cls_infos.keys(), ratios)])

        resampled_cls_num = [int(len(cur_cls_infos) * ratio) for cur_cls_infos, ratio in
                             zip(list(cls_infos.values()), ratios)]
        print('******resampled_cls_num******: ',
              [{k: v} for k, v in zip(list(cls_infos.keys()), resampled_cls_num)])

        sampled_infos = []

        for cur_cls_infos, ratio in zip(list(cls_infos.values()), ratios):
            sampled_infos += np.random.choice(cur_cls_infos, int(len(cur_cls_infos) * ratio)).tolist()
        print('Total samples after balanced resampling: %s' % (len(sampled_infos)))

        # 仅仅为统计重采样后数据集内不同类别目标的比例
        cls_infos_new = {name: [] for name in self.class_names}
        for info in sampled_infos:
            if info['gt_names'].shape == ():
                name = info['gt_names'].tolist()
                if name in self.class_names:
                    cls_infos_new[name].append(info)
            else:
                for name in set(info['gt_names']):
                    if name in self.class_names:
                        cls_infos_new[name].append(info)
        print('Class_samples nums after resampling: ', {k: len(v) for k, v in cls_infos_new.items()})
        cls_dist_new = {k: len(v) / len(sampled_infos) for k, v in cls_infos_new.items()}
        print('Class_samples distribution: ', cls_dist_new)

        num_object_list = [0] * 5
        for info in sampled_infos:
            if info['gt_names'].size == 0:
                continue
            if len(info['gt_names'].shape) == 0:
                info['gt_names'] = np.array([info['gt_names']])
            num_object = info['gt_names'].shape[0]
            gt_names = info['gt_names']
            pt_nums = info['num_lidar_pts']
            for i in range(num_object):
                if gt_names[i] == 'Car' and pt_nums[i] >= 40:
                    num_object_list[0] += 1
                elif gt_names[i] == 'Bus' and pt_nums[i] >= 50:
                    num_object_list[1] += 1
                elif gt_names[i] == 'Tricycle' and pt_nums[i] >= 40:
                    num_object_list[2] += 1
                elif gt_names[i] == 'Mbike' and pt_nums[i] >= 35:
                    num_object_list[3] += 1
                elif gt_names[i] == 'Pedestrian' and pt_nums[i] >= 30:
                    num_object_list[4] += 1
                else:
                    continue
        print('resampled object dist: ', [{k: v} for k, v in zip(cls_infos.keys(), num_object_list)])

        return sampled_infos

    def samples_append(self, points, gt_boxes, gt_names, cls_names, nums_per_sample, probs):
        for i, cls_name in enumerate(cls_names):
            cur_prob = np.random.rand(1).astype(np.float32).item()
            cur_prob_thresh = probs[i]
            if cur_prob < cur_prob_thresh:
                num_cluster = len(self.append_cluster_index[cls_name])
                mask = np.random.randint(0, num_cluster, nums_per_sample[i])
                for j in range(nums_per_sample[i]):
                    start_index = self.append_cluster_index[cls_name][mask[j]][0]
                    end_index = self.append_cluster_index[cls_name][mask[j]][1]
                    cur_cluster_points = self.append_cluster_points[cls_name][start_index:end_index, :]
                    cur_gt_box = self.append_cluster_labels[cls_name][mask[j]]
                    conflict = self.check_conflict(gt_boxes.copy(), cur_gt_box, points, cur_cluster_points.shape[0])
                    if not conflict:
                        cur_cluster_points = np.concatenate(
                            [cur_cluster_points, np.zeros([end_index - start_index, 1])], axis=-1)
                        points = np.concatenate([points, cur_cluster_points], axis=0)
                        cur_gt_box.shape = (1, -1)
                        gt_boxes = np.concatenate([gt_boxes, cur_gt_box], axis=0)
                        gt_names = np.concatenate([gt_names, np.array([cls_name])])
        return points, gt_boxes, gt_names

    @staticmethod
    def check_conflict(original_gt_boxes, new_gt_box, orignal_points, new_object_pointnum):
        shape0 = original_gt_boxes.shape[0]
        if shape0 == 0:
            return False

        conflict = False

        # 考虑new_gt_box是否在original_gt_boxes内部
        new_gt_box.shape = (1, new_gt_box.shape[0])
        for i in range(shape0):
            new_gt_box_copy = new_gt_box.copy()
            new_gt_box_copy[:, 6] -= original_gt_boxes[i, 6]
            new_gt_box_copy_bev = boxes_2_corner(new_gt_box_copy[:, [0, 1, 3, 4, 6]])[0]
            x_min = original_gt_boxes[i, 0] - 0.5 * original_gt_boxes[i, 3]
            x_max = original_gt_boxes[i, 0] + 0.5 * original_gt_boxes[i, 3]
            y_min = original_gt_boxes[i, 1] - 0.5 * original_gt_boxes[i, 4]
            y_max = original_gt_boxes[i, 1] + 0.5 * original_gt_boxes[i, 4]
            if (x_min < new_gt_box_copy_bev[0][0] < x_max and y_min < new_gt_box_copy_bev[0][1] < y_max) or (
                    x_min < new_gt_box_copy_bev[1][0] < x_max and y_min < new_gt_box_copy_bev[1][1] < y_max) \
                    or (
                    x_min < new_gt_box_copy_bev[2][0] < x_max and y_min < new_gt_box_copy_bev[2][1] < y_max) or (
                    x_min < new_gt_box_copy_bev[3][0] < x_max and y_min < new_gt_box_copy_bev[3][1] < y_max):
                conflict = True
                break

        # 考虑每个original_gt_box顶点是否在new_gt_box内部
        original_gt_boxes.shape = (shape0, -1)
        original_gt_boxes[:, 6] -= new_gt_box[0][6]
        original_gt_boxes_bev = boxes_2_corner(original_gt_boxes[:, [0, 1, 3, 4, 6]])

        x_min = new_gt_box[0][0] - 0.5 * new_gt_box[0][3]
        x_max = new_gt_box[0][0] + 0.5 * new_gt_box[0][3]
        y_min = new_gt_box[0][1] - 0.5 * new_gt_box[0][4]
        y_max = new_gt_box[0][1] + 0.5 * new_gt_box[0][4]

        for i in range(original_gt_boxes_bev.shape[0]):
            box1 = original_gt_boxes_bev[i]
            if (x_min < box1[0][0] < x_max and y_min < box1[0][1] < y_max) or (
                    x_min < box1[1][0] < x_max and y_min < box1[1][1] < y_max) \
                    or (x_min < box1[2][0] < x_max and y_min < box1[2][1] < y_max) or (
                    x_min < box1[3][0] < x_max and y_min < box1[3][1] < y_max):
                conflict = True
                break

        # 需要考虑新添加的new_gt_box是否框住了原始点云中的一些点，如果是，则考虑丢弃该new_gt_box
        point_num = get_pointnum(orignal_points, new_gt_box[:, 0:3], new_gt_box[:, 3:6], new_gt_box[:, -1:])[0]
        if point_num > new_object_pointnum * 0.2:
            conflict = True
        return conflict

    # def __len__(self):
    #     """函数头不可删改,返回数据集中样本的个数,必须完善返回值"""
    #     return self.num_samples

    # def __getitem__(self, index):
    #     """
    #     函数头不可删改
    #     功能: 根据index返回训练集/验证集/测试集中单个batch的所有内容
    #           返回给loss_computers.py文件loss_computer.loss_compute()方法 和 data_evaluater.py文件data_evaluater.record()方法
    #           需要判断index和self.num_batch之间的大小关系,从而决定何时结束迭代
    #     """
    #     if index > self.num_batch:
    #         return None, None, None
    #
    #     # 函数返回值inputs和labels需要转换成GPU计算
    #     # inputs等返回值的数据类型并不确定,但应该是由torch.gpu_tensor组成,以方便进行前向传播和损失计算,示例如下
    #     # if torch.cuda.is_available():
    #     #     inputs = Variable(inputs.cuda())
    #     #     labels = Variable(labels.cuda())
    #     # else:
    #     #     self.res_dict['msg'].append('torch.cuda.is_availabel()=False!!!')
    #     #     raise Exceptions
    #
    #     # 返回内容的接口必须固定且必须实现
    #     # inputs直接被传参到 networks.py文件中class Network.forward()方法进行前向传播计算
    #     # labels直接传参到loss_computers.py文件中loss_computer.loss_compute()方法中进行损失计算
    #     # labels还会传参到data_evaluater.py文件中data_evaluater.record()方法中进行性能测评计算
    #     # filenames: batch_size个样本的文件名,以列表的形式组织,len(filenames)=batch_size,用于以同名文件的方式保存每个样本的感知结果
    #     # filenames传参到data_postprocessor.py文件中的data_postprocess()函数中
    #     return inputs, labels, filenames
