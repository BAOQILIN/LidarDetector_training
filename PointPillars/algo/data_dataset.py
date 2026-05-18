#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 18:39:04 2020

@author: lz
"""
import os
import pickle

import numpy as np
import torch
from torch.utils import data

from utils import *


class dataset(data.Dataset):
    """
    Sample-level dataset. Batch assembly is handled in collate_batch().
    """

    def __init__(self, params_dict, folder, prefix, res_dict):
        super(dataset, self).__init__()
        self.params_dict = params_dict
        self.folder = folder
        self.prefix = prefix
        self.res_dict = res_dict

        self.class_names = ['Pedestrian', 'Mbike', 'Car', 'Bus', 'Tricycle']
        self.batch_size = params_dict['TRAIN']['CTRL']['DATA']['BATCH_SIZE'][0]

        if prefix == self.params_dict['TRAIN']['OVERALL']['TRAIN_PREFIX'][0]:
            self.dataset_cfg_path = os.path.join(folder, 'training.pkl')
        elif prefix == self.params_dict['VALIDATION']['VALI_']['PREFIX'][0]:
            self.dataset_cfg_path = os.path.join(folder, 'validation.pkl')
        elif prefix == self.params_dict['TEST']['TEST_']['PREFIX'][0]:
            self.dataset_cfg_path = os.path.join(folder, 'testing.pkl')
        else:
            self.dataset_cfg_path = os.path.join(folder, 'prediction.pkl')

        self.infos = []
        self.include_data(self.dataset_cfg_path)
        if len(self.infos) == 0:
            self.res_dict['msg'].append('Number of samples equals zero!')

        if self.prefix == self.params_dict['TRAIN']['OVERALL']['TRAIN_PREFIX'][0] and \
                self.params_dict['TRAIN']['CTRL']['DATA']['RESAMPLING'][0]:
            self.infos = self.balanced_infos_resampling(self.infos)

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
        self.max_num_points = self.params_dict['TRAIN']['CTRL']['DATA']['MAX_POINTS_PER_VOXEL'][0]
        self.max_voxels = self.params_dict['TRAIN']['CTRL']['DATA']['MAX_VOXEL_NUM'][0]

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

        if self.prefix == self.params_dict['TRAIN']['OVERALL']['TRAIN_PREFIX'][0] \
                and self.params_dict['TRAIN']['CTRL']['AUGMENT']['FILTER_MIN_POINTS'][0]:
            min_points_dict = self.params_dict['TRAIN']['CTRL']['AUGMENT']['MIN_POINTS_NUM'][0]
            self.min_points_each_cls = []
            for cls in self.class_names:
                self.min_points_each_cls.append(min_points_dict[cls.upper()][0])

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        sample = self.infos[index]
        lidar_points_path = sample['lidar_path']
        gt_boxes = sample['gt_boxes'].astype(np.float32)
        gt_names = sample['gt_names']
        num_lidar_pts = sample['num_lidar_pts']

        points = load_ascii_pcd_points(lidar_points_path)

        if self.prefix == self.params_dict['TRAIN']['OVERALL']['TRAIN_PREFIX'][0]:
            if self.params_dict['TRAIN']['CTRL']['AUGMENT']['FILTER_MIN_POINTS'][0]:
                lidar_pts_each_cls = [num_lidar_pts[gt_names == cls] for cls in self.class_names]
                gt_names_each_cls = [gt_names[gt_names == cls] for cls in self.class_names]
                gt_boxes_each_cls = [gt_boxes[gt_names == cls] for cls in self.class_names]
                mask = [lidar_pts_each_cls[i] > self.min_points_each_cls[i] - 1 for i in range(len(self.class_names))]

                gt_names = [gt_names_each_cls[i][mask[i]] for i in range(len(self.class_names))]
                gt_boxes = [gt_boxes_each_cls[i][mask[i]] for i in range(len(self.class_names))]
                gt_names = np.concatenate(gt_names) if len(gt_names) else np.array([])
                gt_boxes = np.concatenate(gt_boxes, axis=0) if len(gt_boxes) else np.zeros((0, 7), dtype=np.float32)

            if self.params_dict['TRAIN']['CTRL']['AUGMENT']['EXTRACT_MINOR_TYPE'][0]:
                points, gt_boxes, gt_names = self.samples_append(
                    points,
                    gt_boxes,
                    gt_names,
                    self.params_dict['TRAIN']['CTRL']['AUGMENT']['APPEND_TYPES'][0],
                    self.params_dict['TRAIN']['CTRL']['AUGMENT']['APPEND_NUM'][0],
                    self.params_dict['TRAIN']['CTRL']['AUGMENT']['APPEND_PROBS'][0],
                )

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

            if gt_boxes.shape[0] > 0:
                gt_boxes[:, 6] = limit_heading(gt_boxes[:, 6], 0.5, 2 * np.pi)
                gt_boxes_mask = np.array([n in self.class_names for n in gt_names], dtype=np.bool_)
                gt_boxes = gt_boxes[gt_boxes_mask]
                gt_names = gt_names[gt_boxes_mask]

        if gt_boxes.shape[0] > 0:
            gt_classes = np.array([self.class_names.index(n) + 1 for n in gt_names], dtype=np.int32)
            gt_boxes = np.concatenate((gt_boxes, gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
        else:
            gt_boxes = np.zeros((0, 8), dtype=np.float32)
        gt_boxes[np.isnan(gt_boxes)] = 0

        nan_mask = None
        for i in range(points.shape[1]):
            if nan_mask is None:
                nan_mask = ~np.isnan(points[:, i])
            else:
                nan_mask *= ~np.isnan(points[:, i])
        points = points[nan_mask]

        return {
            'points': points,
            'gt_boxes': gt_boxes,
            'filename': lidar_points_path.split(os.sep)[-1],
            'frame_id': index,
        }

    @staticmethod
    def collate_batch(batch_list):
        if len(batch_list) == 0:
            return None, None, None

        gt_boxes_list = [sample['gt_boxes'] for sample in batch_list]
        points_list = [sample['points'] for sample in batch_list]
        filename_list = [sample['filename'] for sample in batch_list]
        frame_ids = [sample['frame_id'] for sample in batch_list]

        max_gt = max(gt_box.shape[0] for gt_box in gt_boxes_list) if gt_boxes_list else 0
        gt_dim = gt_boxes_list[0].shape[-1] if gt_boxes_list else 8
        batch_gt_boxes = np.zeros((len(batch_list), max_gt, gt_dim), dtype=np.float32)
        for i, gt_box in enumerate(gt_boxes_list):
            if gt_box.shape[0] > 0:
                batch_gt_boxes[i, :gt_box.shape[0], :] = gt_box

        labels = [
            torch.from_numpy(batch_gt_boxes).float(),
            torch.tensor(frame_ids).float(),
        ]
        return points_list, labels, filename_list

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
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos[name].append(info)

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
                if gt_names[i] == 'Pedestrian' and pt_nums[i] >= 30:
                    num_object_list[0] += 1
                elif gt_names[i] == 'Mbike' and pt_nums[i] >= 35:
                    num_object_list[1] += 1
                elif gt_names[i] == 'Car' and pt_nums[i] >= 40:
                    num_object_list[2] += 1
                elif gt_names[i] == 'Bus' and pt_nums[i] >= 50:
                    num_object_list[3] += 1
                elif gt_names[i] == 'Tricycle' and pt_nums[i] >= 40:
                    num_object_list[4] += 1
        print('Object distribution before resampling: ', [{k: v} for k, v in zip(cls_infos.keys(), num_object_list)])

        original_cls_num = {k: len(v) for k, v in cls_infos.items()}
        print('original_cls_samples: ', original_cls_num)

        duplicated_samples = sum([len(v) for k, v in cls_infos.items()])
        print('Total dupliated samples:', duplicated_samples)

        cls_dist = {k: len(v) / duplicated_samples for k, v in cls_infos.items()}
        print('original cls_samples distribution: ', cls_dist)

        frac = 1.0 / len(self.class_names)
        ratios = [frac / v if v != 0 else 100 for v in cls_dist.values()]
        print('ratios: ', [{k: v} for k, v in zip(cls_infos.keys(), ratios)])

        resampled_cls_num = [int(len(cur_cls_infos) * ratio) for cur_cls_infos, ratio in zip(list(cls_infos.values()), ratios)]
        print('******resampled_cls_num******: ', [{k: v} for k, v in zip(list(cls_infos.keys()), resampled_cls_num)])

        sampled_infos = []
        for cur_cls_infos, ratio in zip(list(cls_infos.values()), ratios):
            sampled_infos += np.random.choice(cur_cls_infos, int(len(cur_cls_infos) * ratio)).tolist()
        print('Total samples after balanced resampling: %s' % (len(sampled_infos)))

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
                if gt_names[i] == 'Pedestrian' and pt_nums[i] >= 30:
                    num_object_list[0] += 1
                elif gt_names[i] == 'Mbike' and pt_nums[i] >= 35:
                    num_object_list[1] += 1
                elif gt_names[i] == 'Car' and pt_nums[i] >= 40:
                    num_object_list[2] += 1
                elif gt_names[i] == 'Bus' and pt_nums[i] >= 50:
                    num_object_list[3] += 1
                elif gt_names[i] == 'Tricycle' and pt_nums[i] >= 40:
                    num_object_list[4] += 1
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
                        cur_cluster_points = np.concatenate([cur_cluster_points, np.zeros([end_index - start_index, 1])], axis=-1)
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
        new_gt_box.shape = (1, new_gt_box.shape[0])
        for i in range(shape0):
            new_gt_box_copy = new_gt_box.copy()
            new_gt_box_copy[:, 6] -= original_gt_boxes[i, 6]
            new_gt_box_copy_bev = boxes_2_corner(new_gt_box_copy[:, [0, 1, 3, 4, 6]])[0]
            x_min = original_gt_boxes[i, 0] - 0.5 * original_gt_boxes[i, 3]
            x_max = original_gt_boxes[i, 0] + 0.5 * original_gt_boxes[i, 3]
            y_min = original_gt_boxes[i, 1] - 0.5 * original_gt_boxes[i, 4]
            y_max = original_gt_boxes[i, 1] + 0.5 * original_gt_boxes[i, 4]
            if (x_min < new_gt_box_copy_bev[0][0] < x_max and y_min < new_gt_box_copy_bev[0][1] < y_max) or \
                    (x_min < new_gt_box_copy_bev[1][0] < x_max and y_min < new_gt_box_copy_bev[1][1] < y_max) or \
                    (x_min < new_gt_box_copy_bev[2][0] < x_max and y_min < new_gt_box_copy_bev[2][1] < y_max) or \
                    (x_min < new_gt_box_copy_bev[3][0] < x_max and y_min < new_gt_box_copy_bev[3][1] < y_max):
                conflict = True
                break

        original_gt_boxes.shape = (shape0, -1)
        original_gt_boxes[:, 6] -= new_gt_box[0][6]
        original_gt_boxes_bev = boxes_2_corner(original_gt_boxes[:, [0, 1, 3, 4, 6]])

        x_min = new_gt_box[0][0] - 0.5 * new_gt_box[0][3]
        x_max = new_gt_box[0][0] + 0.5 * new_gt_box[0][3]
        y_min = new_gt_box[0][1] - 0.5 * new_gt_box[0][4]
        y_max = new_gt_box[0][1] + 0.5 * new_gt_box[0][4]

        for i in range(original_gt_boxes_bev.shape[0]):
            box1 = original_gt_boxes_bev[i]
            if (x_min < box1[0][0] < x_max and y_min < box1[0][1] < y_max) or \
                    (x_min < box1[1][0] < x_max and y_min < box1[1][1] < y_max) or \
                    (x_min < box1[2][0] < x_max and y_min < box1[2][1] < y_max) or \
                    (x_min < box1[3][0] < x_max and y_min < box1[3][1] < y_max):
                conflict = True
                break

        point_num = get_pointnum(orignal_points, new_gt_box[:, 0:3], new_gt_box[:, 3:6], new_gt_box[:, -1:])[0]
        if point_num > new_object_pointnum * 0.2:
            conflict = True
        return conflict
