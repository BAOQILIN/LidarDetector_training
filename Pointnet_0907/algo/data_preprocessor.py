import os
import numpy as np
import json
import pandas as pd
import gc
import yaml


class DataPreprocessor(object):
    def __init__(self, params_dict, data_root, train_flag, res_dict={}):
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
            res_dict: res_dict['msg']是一个列表,列表中的每个元素均为字符串,用户通过向此列表中添加字符串,在web前端页面打印相应的消息
        """
        self.params_dict = params_dict
        self.data_root = data_root
        self.train_flag = train_flag
        self.res_dict = res_dict
        if 'msg' not in self.res_dict:
            self.res_dict['msg'] = []

    def data_preprocess(self):
        """
            数据预处理的唯一固定接口，不可删除，数据预处理过程的具体实现
            原始数据集 --> 预处理完毕的数据集文件(在data_dataset.py中被使用)
        """
        # ...
        # self.func0(param0, param1, ...)
        # self.func1(param0, param1, param2, ...)
        # ...
        scene_list = self.ISceneFilter(self.params_dict, self.data_root, self.res_dict)
        self.res_dict['msg'].append('scene_list: %s' % scene_list)
        print(scene_list)

        num_train, num_vali = self.IPartition(self.params_dict, self.data_root, scene_list, self.res_dict)
        self.res_dict['msg'].append('num_train = %d, num_vali = %d' % (num_train, num_vali))
        print(num_train, num_vali)

        self.ISaveData(self.params_dict, self.data_root, self.res_dict)

        train_frame_num, vali_frame_num = self.IGenClusterData(self.params_dict, self.data_root, self.res_dict)
        self.res_dict['msg'].append('train_frame_num = %d, vali_frame_num = %d' % (train_frame_num, vali_frame_num))
        print('train_frame_num = %d, vali_frame_num = %d' % (train_frame_num, vali_frame_num))

    # def func0(self, arg0, arg1, ...):
    #     """示例：自定义方法0，可删"""
    #
    # def func1(self, arg0, arg1, arg2, ...):
    #     """示例：自定义方法1，可删"""
    #
    # ...
    # "自行增加类内自定义方法"

    def ISceneFilter(self, params_dict, data_root, res_dict={}):
        ori_data = os.path.join(data_root, params_dict["DATASET_FILTER"]["ORI_DATA"][0])
        include_scene = params_dict["DATASET_FILTER"]["INCLUDE_SCENE"][0]
        include_prefix = params_dict["DATASET_FILTER"]["INCLUDE_PREFIX"][0]
        include_suffix = params_dict["DATASET_FILTER"]["INCLUDE_SUFFIX"][0]
        exclude_scene = params_dict["DATASET_FILTER"]["EXCLUDE_SCENE"][0]
        exclude_prefix = params_dict["DATASET_FILTER"]["EXCLUDE_PREFIX"][0]
        exclude_suffix = params_dict["DATASET_FILTER"]["EXCLUDE_SUFFIX"][0]

        scene_list = self.SceneFilter(ori_data,
                                      include_scene, include_prefix, include_suffix,
                                      exclude_scene, exclude_prefix, exclude_suffix,
                                      res_dict=res_dict)
        return scene_list

    def SceneFilter(self,
                    path,
                    include_scene=None, include_prefix=None, include_suffix=None,
                    exclude_scene=None, exclude_prefix=None, exclude_suffix=None,
                    res_dict={}):
        if not os.path.exists(path):
            res_dict['msg'].append("%s is not exists" % (path))

        scene_list = []
        scene_list_tmp = os.listdir(path)

        for scene in scene_list_tmp:
            if not self.CheckScene(scene, include_scene, include_prefix, include_suffix,
                              exclude_scene, exclude_prefix, exclude_suffix):
                continue
            scene_list.append(scene)
        return scene_list

    @staticmethod
    def CheckScene(scene,
                   include_scene=None, include_prefix=None, include_suffix=None,
                   exclude_scene=None, exclude_prefix=None, exclude_suffix=None):
        scene_split = scene.split('_')

        if len(include_scene) != 0 and include_scene[0] != '' and scene not in include_scene:
            return False
        if len(include_prefix) != 0 and include_prefix[0] != '' and scene_split[0] not in include_prefix:
            return False
        if len(include_suffix) != 0 and include_suffix[0] != '' and scene_split[-1] not in include_suffix:
            return False

        if len(exclude_scene) != 0 and exclude_scene[0] != '' and scene in exclude_scene:
            return False
        if len(exclude_prefix) != 0 and exclude_prefix[0] != '' and scene_split[0] in exclude_prefix:
            return False
        if len(exclude_suffix) != 0 and exclude_suffix[0] != '' and scene_split[-1] in exclude_suffix:
            return False

        return True

    def IPartition(self, params_dict, data_root, scene_list, res_dict={}):
        ori_data = os.path.join(data_root, params_dict["PARTITION"]["ORI_DATA"][0])
        output = os.path.join(data_root, params_dict["PARTITION"]["OUTPUT"][0])
        if not os.path.exists(output):
            os.makedirs(output)

        ratio = params_dict["PARTITION"]["RATIO"][0]
        shuffle_index = params_dict["PARTITION"]["SHUFFLE_INDEX"][0]  # 0 random by all cluster; 1 random by all frame; 2 random by all scene
        extra_suffix = params_dict["PARTITION"]["EXTRA_SUFFIX"][0]  # if is not None, it will generate data_train/vali_EXTRA_SUFFIX.json, not data_train.json or data_vali.json
        regenerate = params_dict["PARTITION"]["REGENERATE"][0]  # regenerate file if file is already exists, invalid when extra_suffix is not None

        np.random.seed(params_dict["PARTITION"]["SEED"][0])

        num_train, num_vali = self.Partition(ori_data, scene_list, output, ratio,
                                        shuffle_index, extra_suffix, regenerate, res_dict=res_dict)
        return num_train, num_vali

    # shuffle_index: -1, no random; 0 random by all cluster; 1 random by all frame; 2 random by all scene
    # if extra_suffix is not None, it will generate singular data_train_extra_suffix.json and data_vali_extra_suffix.json, and would not update data_train.json and data_vali.json if they exist already

    def Partition(self, path, scene_list, output, ratio, shuffle_index=0, extra_suffix=None, regenerate=False, res_dict={}):
        all_data = {}

        for scene in scene_list:
            scene_data = self.GetFrameCluster_Onescene(path, scene, res_dict=res_dict)

            if scene_data is not None:
                all_data[scene] = scene_data

        shuffle_data, vali_cluster_end_index = self.ShuffleData(all_data, ratio, shuffle_index)

        if len(extra_suffix) != 0:
            extra_train_file = open(os.path.join(output, 'data_train_' + extra_suffix + '.json'), 'w', encoding='utf-8')
            extra_vali_file = open(os.path.join(output, 'data_vali_' + extra_suffix + '.json'), 'w', encoding='utf-8')

            json.dump(shuffle_data[:vali_cluster_end_index], extra_vali_file, indent=4)
            json.dump(shuffle_data[vali_cluster_end_index:], extra_train_file, indent=4)

            extra_train_file.close()
            extra_vali_file.close()

            if not os.path.exists(os.path.join(output, 'data_train.json')):
                train_file = open(os.path.join(output, 'data_train.json'), 'w', encoding='utf-8')
                json.dump(shuffle_data[vali_cluster_end_index:], train_file, indent=4)
                train_file.close()

            if not os.path.exists(os.path.join(output, 'data_vali.json')):
                vali_file = open(os.path.join(output, 'data_vali.json'), 'w', encoding='utf-8')
                json.dump(shuffle_data[:vali_cluster_end_index], vali_file, indent=4)
                vali_file.close()

        else:

            if regenerate:
                train_file = open(os.path.join(output, 'data_train.json'), 'w', encoding='utf-8')
                vali_file = open(os.path.join(output, 'data_validation.json'), 'w', encoding='utf-8')

                json.dump(shuffle_data[:vali_cluster_end_index], vali_file, indent=4)
                json.dump(shuffle_data[vali_cluster_end_index:], train_file, indent=4)

            else:
                train_num = 0
                vali_num = 0

                if ratio != 0:
                    exists_vali_data = json.load(open(os.path.join(output, 'data_vali.json'), 'r'))

                    vali_file = open(os.path.join(output, 'data_vali.json'), 'w', encoding='utf-8')

                    vali_data = shuffle_data[:vali_cluster_end_index] + exists_vali_data
                    np.random.shuffle(vali_data)

                    json.dump(vali_data, vali_file, indent=4)

                    vali_file.close()

                    vali_num = int(len(vali_data))

                if ratio != 1:
                    exists_train_data = json.load(open(os.path.join(output, 'data_train.json'), 'r'))

                    train_file = open(os.path.join(output, 'data_train.json'), 'w', encoding='utf-8')

                    train_data = shuffle_data[vali_cluster_end_index:] + exists_train_data
                    np.random.shuffle(train_data)

                    json.dump(train_data, train_file, indent=4)

                    train_file.close()

                    train_num = int(len(train_data))

                return train_num, vali_num

        return int(len(shuffle_data) - vali_cluster_end_index), int(vali_cluster_end_index)

    @staticmethod
    def GetFrameCluster_Onescene(path, scene, res_dict={}):

        scene_path = os.path.join(path, scene)
        frame_path = os.path.join(path, scene, 'object_pcd')

        if not os.path.exists(path) or not os.path.exists(scene_path) or not os.path.exists(frame_path):
            res_dict['msg'].append("%s, %s or %s is not exists" % (path, scene_path, frame_path))
            return None
        output = {}
        frame_list = sorted(os.listdir(frame_path))
        for frame in frame_list:
            if not os.path.exists(os.path.join(scene_path, frame + '.txt')):
                continue
            label_file = open(os.path.join(scene_path, frame + '.txt'), 'r')
            output[frame] = []
            for line in label_file.readlines():
                cluster_id = line.split()[17]
                if not os.path.exists(os.path.join(frame_path, frame, cluster_id + '.pcd')):
                    continue
                # output.append([scene, frame, cluster_id])
                output[frame].append(cluster_id)

        return output

    # 0 random by all cluster; 1 random by all frame; 2 random by all scene
    @staticmethod
    def ShuffleData(data, ratio, shuffle_index=0):
        shuffle_data = []
        vali_cluster_end_index = 0
        if shuffle_index == 0:
            for scene in data.keys():
                for frame in data[scene].keys():
                    for cluster in data[scene][frame]:
                        shuffle_data.append({'scene': scene, 'frame': [frame], 'cluster': [[cluster]]})
        if shuffle_index == 1:
            for scene in data.keys():
                for frame in data[scene].keys():
                    cluster = list(data[scene][frame])
                    shuffle_data.append({'scene': scene, 'frame': [frame], 'cluster': [cluster]})
        if shuffle_index == 2:
            for scene in data.keys():
                clusters = []
                frames = list(data[scene].keys())
                for frame in frames:
                    clusters.append(list(data[scene][frame]))
                shuffle_data.append({'scene': scene, 'frame': frames, 'cluster': clusters})

        np.random.shuffle(shuffle_data)
        vali_cluster_end_index = np.ceil(len(shuffle_data) * ratio)

        return shuffle_data, int(vali_cluster_end_index)

    def ISaveData(self, params_dict, data_root, res_dict={}):
        ori_data = os.path.join(data_root, params_dict["DATA_LOAD"]["ORI_DATA"][0])
        partition_data_path = os.path.join(data_root, params_dict["DATA_LOAD"]["PARTITION_DATA_PATH"][0])
        output = os.path.join(data_root, params_dict["DATA_LOAD"]["OUTPUT"][0])
        if not os.path.exists(output):
            os.makedirs(output)

        train_file_prefix = params_dict["DATA_LOAD"]["TRAIN_FILE_PREFIX"][0]
        train_data = os.path.join(partition_data_path, params_dict["DATA_LOAD"]["TRAIN_DATA"][0])
        vali_file_prefix = params_dict["DATA_LOAD"]["VALI_FILE_PREFIX"][0]
        vali_data = os.path.join(partition_data_path, params_dict["DATA_LOAD"]["VALI_DATA"][0])

        self.SaveData(ori_data, train_data, output, train_file_prefix, res_dict=res_dict)
        self.SaveData(ori_data, vali_data, output, vali_file_prefix, res_dict=res_dict)

    def SaveData(self, data_path, data_json, save_path, save_prefix, res_dict={}):

        point_list = []
        data_info = {}
        data_info['scene'] = []
        data_info['frame'] = []
        data_info['cluster'] = []
        data_info['point_num'] = []
        data_info['heading'] = []
        data_info['label'] = []
        data_info['theta'] = []
        data_info['track_id'] = []

        data_none = {}
        data_none['scene'] = []
        data_none['frame'] = []
        data_none['cluster'] = []

        index = -1

        data = self.LoadDataset(data_json, res_dict=res_dict)

        for k in range(len(data)):
            data_ = data[k]
            scene = data_['scene']
            frames = data_['frame']
            clusters = data_['cluster']

            for i in range(len(frames)):
                for j in range(len(clusters[i])):
                    points_ = self.LoadPoints(data_path, scene, frames[i], clusters[i][j], res_dict=res_dict)
                    labels_ = self.LoadLabel(data_path, scene, frames[i], clusters[i][j], res_dict=res_dict)

                    if points_ is None or labels_ is None:
                        data_none['scene'].append(scene)
                        data_none['frame'].append(frames[i])
                        data_none['cluster'].append(clusters[i][j])
                        continue

                    if len(point_list) == 0:
                        point_list = points_
                    else:
                        point_list = np.r_[point_list, points_]

                    data_info['scene'].append(scene)
                    data_info['frame'].append(frames[i])
                    data_info['cluster'].append(clusters[i][j])
                    data_info['point_num'].append(points_.shape[0])
                    data_info['heading'].append(labels_['heading'])
                    data_info['label'].append(labels_['label'])
                    data_info['theta'].append(labels_['theta'])
                    data_info['track_id'].append(labels_['track_id'])

                    index += 1

                    if index % 2000 == 0:
                        res_dict['msg'].append(index)
                    if index % 2000 == 0:
                        print('saving data: %d' % index)

        data_info = pd.DataFrame(data_info)
        data_none = pd.DataFrame(data_none)
        data_info['end_index'] = data_info['point_num'].cumsum()
        data_info['start_index'] = data_info['end_index'] - data_info['point_num']

        # check if already exists
        points_path = os.path.join(save_path, save_prefix + '.npy')
        info_path = os.path.join(save_path, save_prefix + '.csv')
        none_path = os.path.join(save_path, save_prefix + '_none.csv')

        np.save(os.path.join(save_path, save_prefix + '.npy'), point_list)
        data_info.to_csv(os.path.join(save_path, save_prefix + '.csv'), index=False)
        data_none.to_csv(os.path.join(save_path, save_prefix + '_none.csv'), index=False)

        res_dict['msg'].append("success generate " + os.path.join(save_path, save_prefix))

    @staticmethod
    def LoadDataset(path, res_dict={}):
        if not os.path.exists(path):
            res_dict['msg'].append("%s is not exists" % (path))
            return None
        return json.load(open(path))

    @staticmethod
    def LoadPoints(path, scene, frame, cluster, res_dict={}):

        point_path1 = os.path.join(path, scene, 'object_npy', frame, cluster + '.npy')
        point_path2 = os.path.join(path, scene, 'object_pcd', frame, cluster + '.pcd')

        if os.path.exists(point_path1):
            return np.load(point_path1)
        elif os.path.exists(point_path2):
            points_ori = pd.read_csv(point_path2)
            points_num = points_ori.iloc[8][-1].split(' ')[-1]
            if points_num != '0':
                points = pd.read_csv(point_path2, skiprows=11, header=None, sep=' ').values
                os.makedirs(os.path.join(path, scene, 'object_npy', frame))
                np.save(point_path1, points)
                return points
            else:
                return None
        else:
            return None

    @staticmethod
    def LoadLabel(path, scene, frame, cluster, res_dict={}):

        label_file = os.path.join(path, scene, frame + '.txt')
        if not os.path.exists(label_file):
            return None

        cluster_id = int(cluster)

        label_file_handler = open(label_file, 'r')
        lines = label_file_handler.readlines()

        line_index = cluster_id if cluster_id < len(lines) else len(lines) - 1
        pre_line_index = line_index
        line_cluster_id = None

        while line_cluster_id != cluster_id:
            line_split = lines[line_index].split()
            line_cluster_id = int(line_split[17])
            if line_cluster_id == cluster_id:
                if line_split[2] == "Obstacle":
                    label = 0
                elif line_split[2] == "Pedestrian":
                    label = 1
                elif line_split[2] == "Mbike":
                    label = 2
                elif line_split[2] == "Car":
                    label = 3
                elif line_split[2] == "Bus":
                    label = 4
                elif line_split[2] == "Tricycle":
                    label = 5
                elif line_split[2] == "Delete":
                    label = 99
                elif line_split[2] == "Others":
                    label = 999
                else:
                    res_dict['msg'].append(line_split)
                # get heading
                heading = float(line_split[16])
                # get theta
                theta = float(line_split[5])
                # get track id, added in 20200921, liangz
                track_id = int(line_split[1])

                label_file_handler.close()

                return {'label': label, 'heading': heading, 'theta': theta, 'track_id': track_id}

            if line_cluster_id < cluster_id:
                if pre_line_index > line_index:
                    break
                if line_index >= len(lines) - 1:
                    break
                else:
                    pre_line_index = line_index
                    line_index += 1
            if line_cluster_id > cluster_id:
                if pre_line_index < line_index:
                    break
                if line_index <= 0:
                    break
                else:
                    line_index -= 1

        label_file_handler.close()
        return None

    def IGenClusterData(self, params_dict, data_root, res_dict={}):
        npy_data_path = os.path.join(data_root, params_dict["GEN_CLUSTER"]["NPY_DATA_PATH"][0])
        output = os.path.join(data_root, params_dict["GEN_CLUSTER"]["OUTPUT"][0])
        if not os.path.exists(output):
            os.makedirs(output)

        train_data_prefix = params_dict["GEN_CLUSTER"]["TRAIN_DATA_PREFIX"][0]
        train_save_prefix = params_dict["GEN_CLUSTER"]["TRAIN_SAVE_PREFIX"][0]
        vali_data_prefix = params_dict["GEN_CLUSTER"]["VALI_DATA_PREFIX"][0]
        vali_save_prefix = params_dict["GEN_CLUSTER"]["VALI_SAVE_PREFIX"][0]

        min_cluster_points = params_dict["GEN_CLUSTER"]["MIN_CLUSTER_POINTS"][0]
        min_cluster_height = params_dict["GEN_CLUSTER"]["MIN_CLUSTER_HEIGHT"][0]

        train_frame_num = self.generate_clusterdata_from_rawdata(npy_data_path,
                                                                 train_data_prefix,
                                                                 output,
                                                                 train_save_prefix,
                                                                 min_cluster_points,
                                                                 min_cluster_height, res_dict=res_dict)

        vali_frame_num = self. generate_clusterdata_from_rawdata(npy_data_path,
                                                                 vali_data_prefix,
                                                                 output,
                                                                 vali_save_prefix,
                                                                 min_cluster_points,
                                                                 min_cluster_height, res_dict=res_dict)
        return train_frame_num, vali_frame_num

    @staticmethod
    def generate_clusterdata_from_rawdata(dataset_dir, dataset_prefix, save_dir, save_prefix,
                                          uniform_axis=[1, 1, 1],
                                          min_cluster_points=20,
                                          min_cluster_height=0.2, res_dict={}):

        if not os.path.exists(dataset_dir) or not os.path.exists(
                os.path.join(dataset_dir, dataset_prefix + '.npy')) or not os.path.exists(
                os.path.join(dataset_dir, dataset_prefix + '.csv')):
            res_dict['msg'].append("Raw dataset is not exists. please check path")
            return

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        point_list = []
        heading_list = []
        file_list = []
        label_list = []

        total_points = np.load(os.path.join(dataset_dir, dataset_prefix + '.npy'))

        total_info = pd.read_csv(os.path.join(dataset_dir, dataset_prefix + '.csv'),
                                 usecols=['scene', 'frame', 'cluster', 'heading', 'label', 'end_index', 'start_index'],
                                 iterator=True)

        chunkSize = 10000
        load_index = 0

        while True:
            try:
                info_sample = total_info.get_chunk(chunkSize)

                for i in info_sample.index:

                    load_index += 1

                    label = info_sample.at[i, 'label']
                    if label >= 99:
                        continue

                    points = total_points[info_sample.at[i, 'start_index']: info_sample.at[i, 'end_index'], :]
                    if points.shape[0] < min_cluster_points:
                        continue

                    points_min = np.min(points, axis=0)
                    points_max = np.max(points, axis=0)
                    if (points_max[2] - points_min[2]) < min_cluster_height:
                        continue

                    # tmp
                    if "augment" in info_sample.at[i, 'scene']:
                        continue

                    center = 0.5 * (points_min + points_max)  # 点云中心
                    uniform_center = np.array(uniform_axis) * center[:3]
                    points[:, :3] -= uniform_center  # 点云相对于中心点的相对坐标
                    points[:, 3] = points[:, 3] / 256.0  # 强度归一化

                    theta = np.arctan2(center[1], center[0])  # 目标点相对于观测点前进方向的偏向角
                    heading = info_sample.at[i, 'heading']

                    rot_z_mat = np.identity(4)
                    rot_z_mat[0, 0] = np.cos(-theta)
                    rot_z_mat[0, 1] = np.sin(-theta)
                    rot_z_mat[1, 0] = -np.sin(-theta)
                    rot_z_mat[1, 1] = np.cos(-theta)

                    points = np.dot(points, rot_z_mat)  # 点云绕Z轴旋转，旋转角度为theta，或者可以理解成目标点前进方向与观测点统一

                    if heading != -1000:
                        heading = heading - theta

                        while heading > np.pi:
                            heading -= 2 * np.pi
                        while heading < -np.pi:
                            heading += 2 * np.pi

                    file_out = str(info_sample.at[i, 'scene']) + '_' + str(info_sample.at[i, 'frame']) + '_' + str(
                        info_sample.at[i, 'cluster']) + ' ' + str(-theta) + ' ' + str(info_sample.at[i, 'heading'])

                    point_list.append(points[np.random.choice(len(points), 1024, replace=True), 0: 4])  # random sample
                    heading_list.append(heading)
                    file_list.append(file_out)
                    label_list.append(label)

            except StopIteration:
                # self.res_dict['msg'].append("finish loading dataset")
                break

            res_dict['msg'].append("loading dataset: %d" % (load_index))

        # save To numpy
        np.save(os.path.join(save_dir, save_prefix + '_points.npy'), np.array(point_list))
        np.save(os.path.join(save_dir, save_prefix + '_headings.npy'), np.array(heading_list))
        np.save(os.path.join(save_dir, save_prefix + '_label.npy'), np.array(label_list))

        with open(os.path.join(save_dir, save_prefix + '_files.txt'), 'w') as f:
            f.write('\n'.join(list(np.array(file_list))))
        f.close()

        num = int(len(point_list))

        del total_info, total_points, point_list, heading_list, file_list, label_list
        gc.collect()

        return num

# def function0(arg0, arg1, ...):
#     """示例：自定义类外函数0，可删"""
#
# ...
# "自行增加类外自定义函数"


if __name__ == '__main__':
    cfg_file_path = r'D:\Web_LidarDetector\Pointnet\algo\timestamp\algo_config.yaml'
    params_dict = yaml.load(open(cfg_file_path, encoding='utf-8'), Loader=yaml.FullLoader)
    data_root = r'D:\HBOX_Project\LidarDataset\main_dataset\R80_UrbanRoad_20210731_120612'
    res_dict = {'msg': []}
    data_preprocessor = DataPreprocessor(params_dict, data_root, res_dict)
    data_preprocessor.data_preprocess()
