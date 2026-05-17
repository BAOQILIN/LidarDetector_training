import os
import random
import json
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import yaml
from tqdm import tqdm

from utils import load_ascii_pcd_points


def _mask_points_by_range(points, limit_range):
    return (
        (points[:, 0] > limit_range[0]) & (points[:, 0] <= limit_range[3]) &
        (points[:, 1] > limit_range[1]) & (points[:, 1] <= limit_range[4])
    )


def _parse_hx_label_file(label_path, class_mapping):
    with open(label_path, 'r', encoding='utf-8') as f:
        label_dict = json.load(f)

    moving_objects = label_dict.get('movingObjects', [])
    gt_boxes = []
    gt_names = []
    skipped_class_counts = {}
    for obj in moving_objects:
        raw_name = obj.get('objectType')
        mapped_name = class_mapping.get(raw_name)
        if mapped_name is None:
            skipped_class_counts[raw_name] = skipped_class_counts.get(raw_name, 0) + 1
            continue

        cuboid3d = obj.get('annotationTool', {}).get('cuboid3D', {})
        if cuboid3d.get('flag') != 1:
            continue
        cuboid_value = cuboid3d.get('value', {})
        position = cuboid_value.get('position')
        extent = cuboid_value.get('cuboidExtent')
        orientation = cuboid_value.get('orientation')
        if position is None or extent is None or orientation is None:
            continue
        if len(position) < 3 or len(extent) < 3 or len(orientation) < 3:
            continue

        yaw = orientation[2]
        values = position[:3] + extent[:3] + [yaw]
        if any(value is None for value in values):
            continue
        if not np.isfinite(np.asarray(values, dtype=np.float32)).all():
            continue

        gt_boxes.append(values)
        gt_names.append(mapped_name)

    if not gt_boxes:
        return np.zeros((0, 7), dtype=np.float32), np.array([], dtype=str), skipped_class_counts
    return np.array(gt_boxes, dtype=np.float32).reshape(-1, 7), np.array(gt_names), skipped_class_counts


def _get_object_pointnum(points, object_center, size, heading):
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
        x_min = -(size[i, 0] / 2)
        x_max = size[i, 0] / 2
        y_min = -(size[i, 1] / 2)
        y_max = size[i, 1] / 2
        z_min = -(size[i, 2] / 2)
        z_max = size[i, 2] / 2
        point_index = np.where(
            (x_min <= points_re_rot[:, 0]) & (points_re_rot[:, 0] <= x_max) &
            (y_min <= points_re_rot[:, 1]) & (points_re_rot[:, 1] <= y_max) &
            (z_min <= points_re_rot[:, 2]) & (points_re_rot[:, 2] <= z_max),
            True,
            False
        )
        object_pointnum_list.append(len(points_re_rot[point_index]))
    return np.array(object_pointnum_list, dtype=np.int32)


def _build_hx_sample_worker(point_file, ori_data_dir, point_cloud_range, train_flag, test_flag, count_lidar_points, class_mapping):
    lidar_path = os.path.join(ori_data_dir, point_file)
    result = {'sample': None, 'error': None, 'skipped_class_counts': {}}
    if not os.path.isfile(lidar_path):
        return result

    if not train_flag and not test_flag:
        result['sample'] = {
            'lidar_path': lidar_path,
            'gt_boxes': np.zeros((0, 7), dtype=np.float32),
            'gt_names': np.array([], dtype=str),
            'num_lidar_pts': np.array([], dtype=np.int32),
        }
        return result

    stem, _ = os.path.splitext(point_file)
    label_path = os.path.join(ori_data_dir, stem + '.json')
    if not os.path.exists(label_path):
        return result

    try:
        points = load_ascii_pcd_points(lidar_path)
    except ValueError as exc:
        result['error'] = str(exc)
        return result

    points = points[_mask_points_by_range(points, point_cloud_range)]
    gt_boxes, gt_names, skipped_class_counts = _parse_hx_label_file(label_path, class_mapping)
    result['skipped_class_counts'] = skipped_class_counts
    if gt_boxes.shape[0] == 0:
        return result

    if count_lidar_points:
        num_lidar_pts = _get_object_pointnum(points.copy(), gt_boxes[:, :3], gt_boxes[:, 3:6], gt_boxes[:, -1:])
    else:
        num_lidar_pts = np.zeros(gt_boxes.shape[0], dtype=np.int32)

    mask = _mask_points_by_range(gt_boxes[:, :3], point_cloud_range)
    gt_boxes = gt_boxes[mask]
    gt_names = gt_names[mask]
    num_lidar_pts = num_lidar_pts[mask]
    if gt_boxes.shape[0] == 0:
        return result

    result['sample'] = {
        'lidar_path': lidar_path,
        'gt_boxes': gt_boxes,
        'gt_names': gt_names,
        'num_lidar_pts': num_lidar_pts.astype(np.int32),
    }
    return result


class DataPreprocessor(object):
    def __init__(self, params_dict, data_root, train_flag, test_flag, res_dict=None):
        self.params_dict = params_dict
        self.data_root = data_root
        self.train_flag = train_flag
        self.test_flag = test_flag
        self.res_dict = res_dict if res_dict is not None else {}
        if 'msg' not in self.res_dict:
            self.res_dict['msg'] = []

        self.dataset_style = self.params_dict.get('DATASET_STYLE', ['old_platform'])[0]
        self.count_lidar_points = self.params_dict.get('COUNT_LIDAR_POINTS', [True])[0]
        self.use_multiprocess = self.params_dict.get('USE_MULTIPROCESS', [False])[0]
        self.num_workers = self.params_dict.get('NUM_WORKERS', [4])[0]
        self.class_mapping = self._build_class_mapping()
        self.skipped_class_counts = {}

        self.ori_data_dir = os.path.join(self.data_root, self.params_dict['ORI_DATA_PATH'][0])
        self.ori_label_dir = os.path.join(self.data_root, self.params_dict.get('ORI_LABEL_PATH', ['Label'])[0])
        self.label_template_path = os.path.join(self.ori_label_dir, 'label.json')
        self.rectified_data_dir = os.path.join(self.data_root, self.params_dict['SAVE_DATA_PATH'][0])
        if not os.path.exists(self.rectified_data_dir):
            os.makedirs(self.rectified_data_dir)

        self.dataset_list = []

    def data_preprocess(self):
        if self.dataset_style == 'hx_flat':
            self._preprocess_hx_flat()
        else:
            self._preprocess_old_platform()

    def _preprocess_old_platform(self):
        id_2_name_map = {}
        if self.train_flag or self.test_flag:
            if not os.path.exists(self.label_template_path):
                self.res_dict['msg'].append('Error, No label.json!')
                return
            with open(self.label_template_path, 'r', encoding='utf-8') as f:
                temp = json.load(f)
            for each in temp:
                if each['task'] == '目标检测':
                    label_json = each['label']
                    for obj in label_json:
                        id_2_name_map[obj['id']] = obj['name']

        point_cloud_range = self._point_cloud_range()

        self.res_dict['msg'].append('Starting preprocessing data, it may take several minutes...')
        point_file_list = sorted(os.listdir(self.ori_data_dir))
        label_file_set = set(os.listdir(self.ori_label_dir)) if os.path.exists(self.ori_label_dir) else set()
        for point_file in tqdm(point_file_list):
            lidar_path = os.path.join(self.ori_data_dir, point_file)
            if not os.path.isfile(lidar_path):
                continue
            if not point_file.lower().endswith('.pcd'):
                continue

            if not self.train_flag and not self.test_flag:
                self.dataset_list.append({
                    'lidar_path': lidar_path,
                    'gt_boxes': np.zeros((0, 7), dtype=np.float32),
                    'gt_names': np.array([], dtype=str),
                    'num_lidar_pts': np.array([], dtype=np.int32)
                })
                continue

            name, _ = os.path.splitext(point_file)
            label_file = name + '.json'
            if label_file not in label_file_set:
                continue

            try:
                points = load_ascii_pcd_points(lidar_path)
            except ValueError as exc:
                self.res_dict['msg'].append(str(exc))
                continue
            points = points[self.mask_points_by_range(points, point_cloud_range)]
            gt_boxes, gt_names = self._parse_old_platform_label(
                os.path.join(self.ori_label_dir, label_file), id_2_name_map
            )
            if gt_boxes.shape[0] == 0:
                continue

            num_lidar_pts = self._build_num_lidar_pts(points, gt_boxes)
            mask = self.mask_points_by_range(gt_boxes[:, :3], point_cloud_range)
            gt_boxes = gt_boxes[mask]
            gt_names = gt_names[mask]
            num_lidar_pts = num_lidar_pts[mask]
            if gt_boxes.shape[0] == 0:
                continue

            self.dataset_list.append({
                'lidar_path': lidar_path,
                'gt_boxes': gt_boxes,
                'gt_names': gt_names,
                'num_lidar_pts': num_lidar_pts
            })

        self._save_dataset_list()

    def _preprocess_hx_flat(self):
        point_cloud_range = self._point_cloud_range()

        self.res_dict['msg'].append('Starting HX preprocessing data, it may take several minutes...')
        point_file_list = sorted(
            file_name for file_name in os.listdir(self.ori_data_dir)
            if file_name.lower().endswith('.pcd')
        )
        if self.use_multiprocess:
            self._preprocess_hx_flat_multiprocess(point_file_list, point_cloud_range)
        else:
            self._preprocess_hx_flat_single_process(point_file_list, point_cloud_range)
        self._save_dataset_list()

    def _preprocess_hx_flat_single_process(self, point_file_list, point_cloud_range):
        for point_file in tqdm(point_file_list):
            result = self._build_hx_sample(point_file, point_cloud_range)
            self._consume_hx_result(result)

    def _preprocess_hx_flat_multiprocess(self, point_file_list, point_cloud_range):
        worker_count = max(1, int(self.num_workers))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            iterator = executor.map(
                _build_hx_sample_worker,
                point_file_list,
                [self.ori_data_dir] * len(point_file_list),
                [point_cloud_range] * len(point_file_list),
                [self.train_flag] * len(point_file_list),
                [self.test_flag] * len(point_file_list),
                [self.count_lidar_points] * len(point_file_list),
                [self.class_mapping] * len(point_file_list),
            )
            for result in tqdm(iterator, total=len(point_file_list)):
                self._consume_hx_result(result)

    def _build_hx_sample(self, point_file, point_cloud_range):
        return _build_hx_sample_worker(
            point_file,
            self.ori_data_dir,
            point_cloud_range,
            self.train_flag,
            self.test_flag,
            self.count_lidar_points,
            self.class_mapping,
        )

    def _consume_hx_result(self, result):
        if result['error'] is not None:
            self.res_dict['msg'].append(result['error'])
            return
        for class_name, count in result['skipped_class_counts'].items():
            self.skipped_class_counts[class_name] = self.skipped_class_counts.get(class_name, 0) + count
        if result['sample'] is not None:
            self.dataset_list.append(result['sample'])

    def _save_dataset_list(self):
        num_samples = len(self.dataset_list)
        self.res_dict['msg'].append('Total sample number: %s' % num_samples)
        if self.skipped_class_counts:
            self.res_dict['msg'].append('Skipped unmapped HX classes: %s' % self.skipped_class_counts)

        if self.train_flag:
            num_train = int(num_samples * self.params_dict['RATIO'][0])
            random.shuffle(self.dataset_list)
            with open(os.path.join(self.rectified_data_dir, 'training.pkl'), 'wb') as f:
                pickle.dump(self.dataset_list[:num_train], f)
            with open(os.path.join(self.rectified_data_dir, 'validation.pkl'), 'wb') as f:
                pickle.dump(self.dataset_list[num_train:], f)
        elif self.test_flag:
            with open(os.path.join(self.rectified_data_dir, 'testing.pkl'), 'wb') as f:
                pickle.dump(self.dataset_list, f)
        else:
            with open(os.path.join(self.rectified_data_dir, 'prediction.pkl'), 'wb') as f:
                pickle.dump(self.dataset_list, f)

        self.res_dict['msg'].append('Data preprocess done!')

    def _build_class_mapping(self):
        raw_mapping = self.params_dict.get('CLASS_MAPPING', {})
        class_mapping = {}
        for raw_name, mapped_value in raw_mapping.items():
            if isinstance(mapped_value, list):
                class_mapping[raw_name] = mapped_value[0]
            else:
                class_mapping[raw_name] = mapped_value
        return class_mapping

    def _point_cloud_range(self):
        roi = self.params_dict['ROI']
        dims = ['X', 'Y', 'Z']
        point_cloud_range = []
        for dim in dims:
            point_cloud_range.append(roi[f'{dim}_MIN'][0])
        for dim in dims:
            point_cloud_range.append(roi[f'{dim}_MAX'][0])
        return point_cloud_range

    def _parse_old_platform_label(self, label_path, id_2_name_map):
        with open(label_path, 'r', encoding='utf-8') as f:
            label_dict = json.load(f)

        label_object3d = []
        for label in label_dict:
            if label.get('task') == '目标检测':
                label_object3d = label.get('annotation', {}).get('annotation', [])
                break

        gt_boxes = []
        gt_names = []
        for object_label in label_object3d:
            label_id = object_label.get('label_id')
            label_name = id_2_name_map.get(label_id)
            coord = object_label.get('position')
            sizes = object_label.get('dimension')
            rotation = object_label.get('rotation')
            if label_name is None or coord is None or sizes is None or rotation is None:
                continue
            if len(coord) < 3 or len(sizes) < 3 or len(rotation) < 3:
                continue
            gt_boxes.append(coord[:3] + sizes[:3] + [rotation[2]])
            gt_names.append(label_name)

        if not gt_boxes:
            return np.zeros((0, 7), dtype=np.float32), np.array([], dtype=str)
        return np.array(gt_boxes, dtype=np.float32).reshape(-1, 7), np.array(gt_names)

    def _parse_hx_label(self, label_path):
        with open(label_path, 'r', encoding='utf-8') as f:
            label_dict = json.load(f)

        moving_objects = label_dict.get('movingObjects', [])
        gt_boxes = []
        gt_names = []
        for obj in moving_objects:
            raw_name = obj.get('objectType')
            mapped_name = self.class_mapping.get(raw_name)
            if mapped_name is None:
                self.skipped_class_counts[raw_name] = self.skipped_class_counts.get(raw_name, 0) + 1
                continue

            cuboid3d = obj.get('annotationTool', {}).get('cuboid3D', {})
            if cuboid3d.get('flag') != 1:
                continue
            cuboid_value = cuboid3d.get('value', {})
            position = cuboid_value.get('position')
            extent = cuboid_value.get('cuboidExtent')
            orientation = cuboid_value.get('orientation')
            if position is None or extent is None or orientation is None:
                continue
            if len(position) < 3 or len(extent) < 3 or len(orientation) < 3:
                continue

            yaw = orientation[2]
            values = position[:3] + extent[:3] + [yaw]
            if any(value is None for value in values):
                continue
            if not np.isfinite(np.asarray(values, dtype=np.float32)).all():
                continue

            gt_boxes.append(values)
            gt_names.append(mapped_name)

        if not gt_boxes:
            return np.zeros((0, 7), dtype=np.float32), np.array([], dtype=str)
        return np.array(gt_boxes, dtype=np.float32).reshape(-1, 7), np.array(gt_names)

    def _build_num_lidar_pts(self, points, gt_boxes):
        if not self.count_lidar_points:
            return np.zeros(gt_boxes.shape[0], dtype=np.int32)
        return self.get_object_pointnum(points, gt_boxes[:, :3], gt_boxes[:, 3:6], gt_boxes[:, -1:]).astype(np.int32)

    @staticmethod
    def get_object_pointnum(points, object_center, size, heading):
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
            x_min = -(size[i, 0] / 2)
            x_max = size[i, 0] / 2
            y_min = -(size[i, 1] / 2)
            y_max = size[i, 1] / 2
            z_min = -(size[i, 2] / 2)
            z_max = size[i, 2] / 2
            point_index = np.where(
                (x_min <= points_re_rot[:, 0]) & (points_re_rot[:, 0] <= x_max) &
                (y_min <= points_re_rot[:, 1]) & (points_re_rot[:, 1] <= y_max) &
                (z_min <= points_re_rot[:, 2]) & (points_re_rot[:, 2] <= z_max),
                True,
                False
            )
            object_pointnum_list.append(len(points_re_rot[point_index]))
        return np.array(object_pointnum_list)

    @staticmethod
    def mask_points_by_range(points, limit_range):
        mask = (
            (points[:, 0] > limit_range[0]) & (points[:, 0] <= limit_range[3]) &
            (points[:, 1] > limit_range[1]) & (points[:, 1] <= limit_range[4])
        )
        return mask


if __name__ == '__main__':
    cfg_file_path = r'D:\Web_LidarDetector\Pointnet\algo\timestamp\algo_config.yaml'
    params_dict = yaml.load(open(cfg_file_path, encoding='utf-8'), Loader=yaml.FullLoader)
    data_root = r'D:\HBOX_Project\LidarDataset\main_dataset\R80_UrbanRoad_20210731_120612'
    train_flag = True
    res_dict = {'msg': []}
    data_preprocessor = DataPreprocessor(params_dict, data_root, train_flag, False, res_dict)
    data_preprocessor.data_preprocess()
