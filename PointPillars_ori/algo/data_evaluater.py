import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from DataUtils.data_evaluater_base import data_evaluater_base
import utils
import os
import pandas as pd
import numba
import copy
from munkres import Munkres

import matplotlib.pyplot as plt


# class data_evaluater_base(metaclass=abc.ABCMeta):
#     def __init__(self, params_dict, result_root, train_flag, res_dict={}):
#         self.params_dict = params_dict
#         self.result_root = result_root
#         self.train_flag = train_flag
#         self.res_dict = res_dict
#         if 'msg' not in self.res_dict:
#             self.res_dict['msg'] = []

#         self.result_path = os.path.join(self.result_root, self.params_dict['TRAIN']['PATH']['RESULT_PATH'][0])
#         if not os.path.exists(self.result_path):
#             os.makedirs(self.result_path)

#         self.evaluate_results = []

#         if self.train_flag:
#             self.evaluate_file = 'eva_vali.npy'
#         else:
#             self.evaluate_file = 'eva_test.npy'

#         if os.path.exists(os.path.join(self.result_path, self.evaluate_file)):
#             self.evaluate_results = np.load(os.path.join(self.result_path, self.evaluate_file),
#                                                                 allow_pickle=True).item()['evaluate_result']
#         if self.params_dict['TRAIN']['OVERALL']['INITIAL_RESULT'][0]:
#             self.evaluate_results = []

#     def save(self):
#         np.save(os.path.join(self.result_path, self.evaluate_file), {'evaluate_result': self.evaluate_results})

#     #    @abc.abstractmethod
#     def initial(self, dataset_type='eva_train'):
#         pass

#     #    @abc.abstractmethod
#     def record(self, model_outputs, true):
#         pass

#     #    @abc.abstractmethod
#     def evaluate(self):
#         pass


class data_evaluater(data_evaluater_base):
    def __init__(self, params_dict, result_root, train_flag, res_dict={}):
        """
        函数头与函数体不可删改,函数体可自定义增加内容
        功能: 接收模型预测结果和真值标签进行评价,保存本地
        :param params_dict: 解析algo_config.yaml中的['TRAIN_MODEL']部分得到的参数字典,具体内容用户可自定义
        :param result_root: 测评结果文件保存的根目录,由系统提供,用户不可见,具体使用见父类save()方法
        :param train_flag: True/False,说明当前是处于模型训练过程 or 模型测试过程
                           模型训练过程则是对验证集进行测评,并将结果保存到eva_vali.npy文件中;
                           模型测试过程则是对测试集进行测评,并将结果保存到eva_test.npy文件中;
                           见基类self.evaluate_file定义
        :param res_dict: 字典结构,res_dict['msg']为列表,列表中的元素为字符串,前端将显示打印其中的所有元素,用户可自行向其中添加字符串
        """
        super().__init__(params_dict, result_root, train_flag, res_dict=res_dict)
        self.preds = None
        self.gts = None

        self.class_names = np.array(['Pedestrian', 'Mbike', 'Car', 'Bus', 'Tricycle'])
        self.initial()

    def initial(self):
        """函数头不可更改, 后续函数体可自定义"""
        # 建议初始化如下成员变量,用于保存整个数据集的真值和预测值,可自定义
        self.preds = None
        self.labels = None

    def record(self, model_outputs, labels):
        """
        函数头不可改动,函数体自定义
        功能: 接受单个batch的模型输出预测值和对应的标签真值,经过相应处理后添加到self.labels和self.preds(具体形式可自定义)
        model_outputs: 由model_computers.py文件中 class model_computer_base.model_compute()方法得到的单个batch模型预测值
                       同时也是networks.py文件中 class Network.forward()方法的返回值
        labels: 对应batch的真值,data_dataset.py文件中class dataset.__getitem__()方法中的返回值labels
        """
        batch_cls_preds = model_outputs['cls_preds']  # list:5 (shape=(b, 128*128*2*C', C') C'代表该检测头所对应的类别个数
        batch_box_preds = model_outputs['box_preds']  # list:5 (shape=(b, 128*128*2*C', 8) C'代表检测头所对应的的类别个数
        batch_size = batch_cls_preds[0].shape[0]
        gt_boxes, frame_ids = labels
        gt_boxes = gt_boxes.cpu().numpy()
        frame_ids = frame_ids.cpu().numpy()

        self.params_dict_copy = copy.deepcopy(self.params_dict)
        bbox_generator = utils.BboxGenerator(self.params_dict_copy)
        pred_dicts = bbox_generator.generate_predicted_boxes(batch_size, batch_cls_preds, batch_box_preds)
        for index, box_dict in enumerate(pred_dicts):
            pred_scores = box_dict['pred_scores'].cpu().numpy()[:, np.newaxis]
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            num_preds = pred_scores.shape[0]
            if num_preds != 0:
                pred_labels = self.class_names[box_dict['pred_labels'].cpu().numpy() - 1][:, np.newaxis]


                cur_pred = np.concatenate([(np.ones([num_preds, 1]) * frame_ids[index].item()).astype(np.int16),
                                           -np.ones([num_preds, 1]).astype(np.int8),
                                           pred_labels,
                                           -np.ones([num_preds, 2]).astype(np.int8),
                                           -1000 * np.ones([num_preds, 5]).astype(np.int8),
                                           pred_boxes[:, 5:6], pred_boxes[:, 4:5], pred_boxes[:, 3:4],
                                           pred_boxes[:, :3], pred_boxes[:, 6:7],
                                           np.arange(num_preds).reshape(num_preds, 1).astype(np.int8),
                                           pred_scores, -1000 * np.ones([num_preds, 4]).astype(np.int8)], axis=-1)

                if self.preds is None:
                    self.preds = cur_pred
                else:
                    print('self.preds.shape[0]: %i, current_pred.shape[0]: %i' % (self.preds.shape[0], cur_pred.shape[0]))
                    self.preds = np.concatenate([self.preds, cur_pred], axis=0)

            gt_box = gt_boxes[index]
            cur_index = gt_box.shape[0] - 1
            while cur_index > 0 and np.sum(gt_box[cur_index]) == 0:
                cur_index -= 1
            gt_box = gt_box[:cur_index]
            num_gt = gt_box.shape[0]
            if num_gt != 0:
                gt_labels = self.class_names[(gt_box[:, -1] - 1).astype(np.int)][:, np.newaxis]
                cur_gt = np.concatenate([(np.ones([num_gt, 1]) * frame_ids[index].item()).astype(np.int16),
                                         -np.ones([num_gt, 1]).astype(np.int8),
                                         gt_labels,
                                         -np.ones([num_gt, 2]).astype(np.int8),
                                         -1000 * np.ones([num_gt, 5]).astype(np.int8),
                                         gt_box[:, 5:6], gt_box[:, 4:5], gt_box[:, 3:4],
                                         gt_box[:, :3], gt_box[:, 6:7],
                                         np.arange(num_gt).reshape(num_gt, 1).astype(np.int8),
                                         np.ones([num_gt, 1]).astype(np.int8),
                                         -1000 * np.ones([num_gt, 4]).astype(np.int8)], axis=-1)
                if self.gts is None:
                    self.gts = cur_gt
                else:
                    self.gts = np.concatenate([self.gts, cur_gt], axis=0)

    def evaluate(self):
        """
        函数头不可改动,函数体自定义
        功能: 对整个数据集的预测值self.preds和真值self.labels进行测评,得到相关指标(eg:AP、precision、recall.etc)并保存
        """

        # 测评的指标结果必须保存到self.evaluate_results列表中
        # 代码示例如下,列表中元素的具体形式和内容可以自定义
        # 测评结束后父类中save()方法会保存测评结果self.evaluate_results到本地文件eva_vali.npy/eva_test.npy.
        preds = pd.DataFrame(self.preds, columns=['frame', 'track_id', 'obj_type', 'truncation', 'occlusion', 'theta',
                                                  'left', 'top', 'rigth', 'bottom', 'heigth', 'width', 'length',
                                                  'x', 'y', 'z', 'heading', 'cluster_id', 'score', 'v_x', 'v_y', 'v_z',
                                                  'acc'])
        preds[['frame', 'track_id', 'heigth', 'width', 'length', 'x', 'y', 'z', 'heading', 'cluster_id', 'score']] = \
            preds[['frame', 'track_id', 'heigth', 'width', 'length', 'x', 'y', 'z', 'heading', 'cluster_id',
                   'score']].apply(pd.to_numeric, errors='ignore')
        gts = pd.DataFrame(self.gts, columns=['frame', 'track_id', 'obj_type', 'truncation', 'occlusion', 'theta',
                                              'left', 'top', 'rigth', 'bottom', 'heigth', 'width', 'length',
                                              'x', 'y', 'z', 'heading', 'cluster_id', 'score', 'v_x', 'v_y', 'v_z',
                                              'acc'])
        gts[['frame', 'track_id', 'heigth', 'width', 'length', 'x', 'y', 'z', 'heading', 'cluster_id', 'score']] = \
            gts[['frame', 'track_id', 'heigth', 'width', 'length', 'x', 'y', 'z', 'heading', 'cluster_id',
                 'score']].apply(pd.to_numeric, errors='ignore')

        self.gts = None
        self.preds = None

        if preds.shape[0] == 0:
            self.evaluate_results.append(
                [np.zeros([5]), np.array([np.nan] * 5), np.array([np.nan] * 5),
                 0.0 * np.ones([5])])  # {'recall': 0, 'precision': np.nan, 'ap': np.nan, 'aos': 0.5}
            self.res_dict['msg'].append("No prediction, Evaluation Done!")

        else:
            eva = data_evaluate.trackingEvaluation()
            scene = 'val'
            evaluate_task = 'Detection'
            detection_type = self.class_names.tolist()
            association_mode = 'IOU'
            criterion = ['union', 0.5]

            res = eva.evaluate(scene, gts, preds, evaluate_task, detection_type, association_mode, criterion)
            if res:
                print("evaluation Done!")
                self.res_dict['msg'].append('evaluation Done!')

                recall = np.mean(
                    eva.detect_res[scene]['recall'].loc[:, ['Car', 'Bus', 'Tricycle', 'Mbike', 'Pedestrian']]).values
                precision = np.mean(
                    eva.detect_res[scene]['precision'].loc[:, ['Car', 'Bus', 'Tricycle', 'Mbike', 'Pedestrian']]).values
                ap = np.mean(
                    eva.detect_res[scene]['ap'].loc[:, ['Car', 'Bus', 'Tricycle', 'Mbike', 'Pedestrian']]).values
                aos = np.mean(
                    eva.detect_res[scene]['aos'].loc[:, ['Car', 'Bus', 'Tricycle', 'Mbike', 'Pedestrian']]).values
                self.evaluate_results.append([recall, precision, ap, aos])

def PlotRec(points, color, set_axis=False):
    if points == []:
        return
    if points.shape[0] == 1:
        plt.scatter(points[:, 0], points[:, 1], c=color)
        return
    if points.shape[0] == 2:
        plt.plot([points[0, 1], points[1, 1]], [points[0, 0], points[1, 0]], '-o', color=color)
        return

    points = np.r_[points, points[0, :].reshape([1, -2])]

    for i in range(1, points.shape[0]):
        plt.plot([points[i - 1, 1], points[i, 1]], [points[i - 1, 0], points[i, 0]], '-o', color=color)

    if set_axis:
        plt.xlabel('y')
        plt.ylabel('x')
        ax = plt.gca()
        ax.invert_xaxis()
        # ax.invert_yaxis()
        ax.yaxis.set_ticks_position('right')
        # ax.set_aspect('equal', adjustable='box')

    return


# pots1: N * dim
# pots2: M * dim
# out: N * M
def ComputeDisByPots(pots1, pots2):
    a = np.sum(np.square(pots1), axis=1)
    a2 = np.repeat(a.reshape([-1, 1]), pots2.shape[0], axis=1)

    b = np.sum(np.square(pots2), axis=1)
    b2 = np.repeat(b.reshape([1, -1]), pots1.shape[0], axis=0)

    dis = a2 + b2 - 2 * np.dot(pots1, pots2.T)
    dis[dis < 0] = 0  # raised by float error

    dis = np.sqrt(dis)

    return dis


# input: N * 19
# output: 4 * 2 * N,  left top, right top, right bottom, left bottom
def ComputeVertexCoor(data):
    heading = data['heading'].values
    heading[heading == -1000] = 0
    rot_mat = np.array([[np.cos(heading), np.sin(heading)], [-np.sin(heading), np.cos(heading)]])  # 2 * 2 * N

    bbox_sizes = data.loc[:, ['length', 'width']].values
    bbox_sizes = bbox_sizes.T.reshape([1, 2, -1])  # 1 * 2 * N

    vertex_bbox_bias = np.array([[0.5, 0.5], [0.5, -0.5], [-0.5, -0.5], [-0.5, 0.5]]).reshape([4, 2, 1])  # 4 * 2 * 1
    vertex_bbox = bbox_sizes * vertex_bbox_bias  # 4 * 2 * N

    bbox_ori_pot = data.loc[:, ['x', 'y']].values  # N * 2
    bbox_ori_pot = bbox_ori_pot.reshape([-1, 1, 2])  # N * 1 * 2
    vertex_lidar = np.matmul(vertex_bbox.transpose([2, 0, 1]), rot_mat.transpose([2, 0, 1]))  # N * 4 * 2
    vertex_lidar = vertex_lidar + bbox_ori_pot  # N * 4 * 2

    return vertex_lidar.transpose([1, 2, 0])


# return point with minimum distance from original point on line: [min_pot, min_pot_side]
def ComputeMinPointOnLine(min_pot, min_pot_side):
    v1 = min_pot_side - min_pot
    v2 = -min_pot
    r = np.dot(v1, v2) / np.sum(np.square(v1))
    if r < 0:
        return min_pot, np.sqrt(np.sum(np.square(min_pot)))
    elif r > 1:
        return min_pot_side, np.sqrt(np.sum(np.square(min_pot_side)))
    else:
        min_pot_online = min_pot + v1 * r
        return min_pot_online, np.sqrt(np.sum(np.square(min_pot_online)))


# input: 4 * 2 * N, row: left top, right top, right bottom, left bottom, col: x, y 
# out: N * 2, row: bbox, col: min pot
def ComputeMinPointInBBox(vertex_lidar):
    vertex_dis = np.sqrt(np.sum(np.square(vertex_lidar), axis=1))
    vertex_min_dis_index = np.argmin(vertex_dis, axis=0)
    output = np.zeros([vertex_dis.shape[1], 2])
    for i in range(output.shape[0]):
        min_index = vertex_min_dis_index[i]
        min_pot = vertex_lidar[min_index, :, i]
        min_pot_side1 = vertex_lidar[(min_index + 1) % 4, :, i]
        min_pot_side2 = vertex_lidar[(min_index - 1) % 4, :, i]
        min_pot1, dis_1 = ComputeMinPointOnLine(min_pot, min_pot_side1)
        min_pot2, dis_2 = ComputeMinPointOnLine(min_pot, min_pot_side2)
        if dis_1 < dis_2:
            output[i, :] = min_pot1
        else:
            output[i, :] = min_pot2

    return output


# pot1, pot2, pot3: [x, y]
def CaluateTriArea(pot1, pot2, pot3):
    a = np.linalg.norm(pot1 - pot2)
    b = np.linalg.norm(pot2 - pot3)
    c = np.linalg.norm(pot3 - pot1)
    s = (a + b + c) * 0.5
    area = np.sqrt(np.clip(s * (s - a) * (s - b) * (s - c), a_min=1e-8, a_max=1e8))
    return area


def CaluateTriArea1(pot1, pot2, pot3):
    a = np.linalg.norm(pot1 - pot2)
    b = np.linalg.norm(pot2 - pot3)
    c = np.linalg.norm(pot3 - pot1)
    s = (a + b + c) * 0.5
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    return area


# pot: [x, y]
# rec_points: N * 2
def PointInRec(pot, rec_points):
    flag = None
    for i in range(rec_points.shape[0]):
        tmp = np.cross(rec_points[(i + 1) % rec_points.shape[0], :] - rec_points[i, :], pot - rec_points[i, :])
        if flag is None:
            flag = tmp
        else:
            if flag * tmp < 0:
                return False

    return True


# line1: [[xa, ya], [xb, yb]]
# line2: [[xc, yc], [xd, yd]]
# ref: https://blog.csdn.net/wcl0617/article/details/78654944
def LineCrossLine(line1, line2):
    ab = line1[1, :] - line1[0, :]
    cd = line2[1, :] - line2[0, :]
    ac = line2[0, :] - line1[0, :]
    ad = line2[1, :] - line1[0, :]

    ab_cross_ac = np.cross(ab, ac)
    ab_cross_ad = np.cross(ab, ad)

    if ab_cross_ac * ab_cross_ad >= 0:
        return []

    ab_cross_cd = np.cross(ab, cd)

    t = np.cross(ac, cd) / ab_cross_cd
    u = np.cross(ab, -ac) / ab_cross_cd

    if t <= 1 and t >= 0 and u >= 0 and u <= 1:
        return list(line1[0, :] + t * ab)
    else:
        return []

    # line: [[xa, ya], [xb, yb]]


# rec: 4 * 2
def LineCrossRec(line, rec):
    rec_pot_num = rec.shape[0] - 1
    # rec_local = np.r_[rec, rec[0, :].reshape([1, 2])]
    cross_points = []

    for i in range(rec_pot_num):
        cross_point = LineCrossLine(line, rec[i:i + 2, :])
        if len(cross_point) != 0:
            cross_points.append(cross_point)

    if len(cross_points) == 1:
        # 把端点也作为多边行的顶点
        if PointInRec(line[0, :], rec):
            cross_points.append(list(line[0, :]))
        else:
            cross_points.append(list(line[1, :]))

    if len(cross_points) == 0 and PointInRec(line[0, :], rec) and PointInRec(line[1, :], rec):
        # 如果一条线与矩形没有交点，需要考虑点在矩形内部
        cross_points.append(list(line[0, :]))
        cross_points.append(list(line[1, :]))

    return cross_points


# rec1: N * 2
# rec2: M * 2
# ref: https://blog.csdn.net/Ghy817920/article/details/85067993
def RecCrossRec(rec1, rec2):
    rec1_num = rec1.shape[0]
    rec2_num = rec2.shape[0]

    rec1_local = np.concatenate((rec1, rec1[0, :].reshape(1, 2)), axis=0)
    rec2_local = np.concatenate((rec2, rec2[0, :].reshape(1, 2)), axis=0)
    cross_points = []

    for i in range(rec1_num):
        cross_point = LineCrossRec(rec1_local[i:i + 2, :], rec2_local)
        if cross_point != []:
            cross_points.extend(cross_point)

    for j in range(rec2_num):
        cross_point = LineCrossRec(rec2_local[j:j + 2, :], rec1_local)
        if cross_point != []:
            cross_points.extend(cross_point)

    if cross_points == []:
        return cross_points, 0

    cross_points = np.array(cross_points)
    new_points = cross_points[0, :].reshape([-1, 2])

    while True:
        cross_points = cross_points[np.sum(np.square(cross_points - new_points[-1, :]), axis=1) > 1e-6]
        if cross_points.shape[0] != 0:
            new_points = np.r_[new_points, cross_points[0, :].reshape([-1, 2])]
        else:
            break

    if new_points.shape[0] <= 2:
        return new_points, 0

    if new_points.shape[0] == 3:
        return new_points, CaluateTriArea(new_points[0, :], new_points[1, :], new_points[2, :])

    centroid = np.mean(new_points, axis=0)
    ori_pot = new_points[0, :]
    co = ori_pot - centroid
    angle = [0]
    for i in range(1, new_points.shape[0]):
        cp = new_points[i, :] - centroid
        angle_i_cos = co.dot(cp) / (np.linalg.norm(co) * np.linalg.norm(cp))
        angle_i_cos = min(1, max(-1, angle_i_cos))  # float error
        angle_i = np.arccos(angle_i_cos)
        cross_i = np.cross(co, cp)
        # > 180
        if cross_i < 0:
            angle.append(2 * np.pi - angle_i)
        else:
            angle.append(angle_i)

    new_points = new_points[np.argsort(angle), :]

    S = 0
    for i in range(1, new_points.shape[0] - 1):
        S += CaluateTriArea(new_points[0, :], new_points[i, :], new_points[i + 1, :])

    return new_points, S


class trackingEvaluation(object):
    """ tracking statistics (CLEAR MOT, id-switches, fragments, ML/PT/MT, precision/recall)
             MOTA	- Multi-object tracking accuracy in [0,100]
             MOTP	- Multi-object tracking precision in [0,100] (3D) / [td,100] (2D)

             id-switches - number of id switches
             fragments   - number of fragmentations

             MT, PT, ML	- number of mostly tracked, partially tracked and mostly lost trajectories
    """

    def __init__(self, obj_type=['Obstacle', 'Pedestrian', 'Mbike', 'Car', 'Bus', 'Tricycle', 'Delete', 'Others']):

        self.groundtruth_data = {}
        self.track_data = {}
        self.obj_type = obj_type

        self.track_res = {}
        self.detect_res = {}

        self.max_MinPoint_Thresh = 1.0
        self.min_IOU_Thresh = 0.2

        self.default_MinPoint_Thresh = 0.5
        self.default_IOU_Thresh = 0.5

    def evaluate(self, scene, gt_data, tr_data, evaluate_type='Detection', detection_type=None,
                 mode='IOU', criterion=['union', 0.5], recheck=False,
                 filter_type='any', ignore_type=None, ignore_occlusion=None, ignore_truncation=None, ignore_radius=None,
                 ignore_xy_range=None):

        if gt_data is None:
            return False

        if tr_data is None:
            return False

        self.groundtruth_data[scene] = gt_data
        self.track_data[scene] = tr_data

        tr_data_filter = tr_data.copy()
        if ignore_type is not None:
            keep_obj_type = [i for i in self.obj_type if i not in ignore_type]
        else:
            keep_obj_type = self.obj_type
        tr_remove_index = self._filterData(tr_data_filter, keep_obj_type, None, None, ignore_radius, ignore_xy_range,
                                           None, filter_type)
        tr_data_filter = tr_data_filter.loc[~tr_remove_index, :].reset_index(drop=True)

        # evaluate
        if evaluate_type == 'Detection' or evaluate_type == 'All':
            recall, precision, fp_rate, ap, aos = self._EvaluateDetection(gt_data, tr_data_filter, detection_type, mode,
                                                                          criterion, recheck,
                                                                          filter_type, ignore_type, ignore_occlusion,
                                                                          ignore_truncation, ignore_radius,
                                                                          ignore_xy_range)
            self.detect_res[scene] = {}
            self.detect_res[scene]['recall'] = recall
            self.detect_res[scene]['precision'] = precision
            self.detect_res[scene]['fp_rate'] = fp_rate
            self.detect_res[scene]['ap'] = ap
            self.detect_res[scene]['aos'] = aos

        if evaluate_type == 'Track' or evaluate_type == 'All':
            res = self._EvaluateTrack(gt_data, tr_data_filter, mode, criterion, recheck,
                                      filter_type, ignore_type, ignore_occlusion, ignore_truncation, ignore_radius,
                                      ignore_xy_range)
            self.track_res[scene] = res

        return True

    def output(self, output_path, file_prefix, scene, evaluate_type='Detection', detection_output_type=None,
               detection_output_metrics=None):

        if detection_output_type is None: detection_output_type = self.obj_type
        if detection_output_metrics is None: detection_output_metrics = ['fp_rate', 'recall', 'precision', 'ap', 'aos']

        if evaluate_type == 'Track' or evaluate_type == 'All':
            track_path = os.path.join(output_path, file_prefix + '_track.csv')
            pd.DataFrame(self.track_res[scene], index=[file_prefix]).to_csv(track_path)

        if evaluate_type == 'Detection' or evaluate_type == 'All':
            detec_path = os.path.join(output_path, file_prefix + '_detection.csv')

            metrics_out = []
            for metrics_type in detection_output_metrics:
                metrics_out.append(np.mean(self.detect_res[scene][metrics_type]))

            detec_out = pd.DataFrame()
            detec_out['index'] = [file_prefix]

            for j in range(len(detection_output_metrics)):
                for i in range(len(detection_output_type)):
                    detec_type = detection_output_type[i]
                    detec_metr = detection_output_metrics[j]

                    detec_out[detec_metr + '_' + detec_type] = [metrics_out[j][detec_type]]

            detec_out.to_csv(detec_path)

    def _loadData(self, path, scene):
        """
            Generic loader for ground truth and tracking data in main_dataset format.
        """

        # if not os.path.exists(path) or not os.path.exists(os.path.join(path, scene)) or not os.path.exists(os.path.join(path, scene, 'object_pcd')):
        if not os.path.exists(path) or not os.path.exists(os.path.join(path, scene)):
            print(os.path.join(path, scene))
            print("path of given scene is not exists")
            return None

        scene_path = os.path.join(path, scene)

        # frame_list = sorted(os.listdir(os.path.join(scene_path, 'object_pcd')))

        # frame_list = []
        # for i in range(140):
        #     tmp = str(i*5)
        #     while len(tmp) < 6:
        #         tmp = "0" + tmp
        #     frame_list.append(tmp)

        temp = os.listdir(os.path.join(path, scene))
        frame_list = []
        for each in temp:
            if each[-4:] == '.pcd':
                frame_list.append(each[:-4])

        scene_data = None

        for frame in frame_list:

            if not os.path.exists(os.path.join(scene_path, frame + '.txt')):
                continue

            label_data = pd.read_csv(os.path.join(scene_path, frame + '.txt'), sep=' ', header=None,
                                     names=['frame', 'track_id', 'obj_type', 'truncation', 'occlusion', 'theta',
                                            'left', 'top', 'rigth', 'bottom', 'heigth', 'width', 'length',
                                            'x', 'y', 'z', 'heading', 'cluster_id', 'score', 'v_x', 'v_y', 'v_z',
                                            'acc'])

            if scene_data is None:
                scene_data = label_data
            else:
                scene_data = scene_data.append(label_data)

        scene_data.reset_index(drop=True, inplace=True)

        return scene_data

    '''
    obj_type: None or list type, remove: type not in obj_type
    occulsion: None or a positive integra value, remove: >= occulsion 
    truncation: None or a float value between 0 ~ 1, remove: >= truncation
    radius: None or a single number, remove: > radius
    xy_range: None or list type, [x_min, x_max, y_min, y_max], remove: not in xy-range
    frame_range: None or list type, [start_frame, end_frame], remove: not in frame-range
    filter_type: 'any': any valid condition satisfied, 'all': all valid condition satisfied
    '''

    def _filterData(self, data, obj_type=None, occulsion=None, truncation=None, radius=None, xy_range=None,
                    frame_range=None, filter_type='all'):

        remove_index = np.array([False for i in range(data.shape[0])])

        if obj_type is not None:
            remove_obj = data.apply(lambda x: x.obj_type not in obj_type, axis=1)
            if filter_type == 'any':
                remove_index = remove_index | remove_obj
            else:
                remove_index = remove_index & remove_obj

        if occulsion is not None:
            remove_occlusion = np.array([False] * data.shape[0])
            if occulsion[0] is not None:
                remove_occlusion = remove_occlusion | (data.occlusion <= occulsion[0])
            if occulsion[1] is not None:
                remove_occlusion = remove_occlusion | (data.occlusion >= occulsion[1])

            if filter_type == 'any':
                remove_index = remove_index | remove_occlusion
            else:
                remove_index = remove_index & remove_occlusion

        if truncation is not None:
            remove_truncation = data.truncation >= truncation
            if filter_type == 'any':
                remove_index = remove_index | remove_truncation
            else:
                remove_index = remove_index & remove_truncation

        if radius is not None:
            remove_radius = ((data.x * data.x + data.y * data.y) > (radius * radius)).values
            if filter_type == 'any':
                remove_index = remove_index | remove_radius
            else:
                remove_index = remove_index & remove_radius

        if xy_range is not None:
            x_filter = np.array([False] * data.shape[0])
            y_filter = np.array([False] * data.shape[0])

            if xy_range[0] is not None:
                x_filter = x_filter | (data.x < xy_range[0])
            if xy_range[1] is not None:
                x_filter = x_filter | (data.x > xy_range[1])
            if xy_range[2] is not None:
                y_filter = y_filter | (data.y < xy_range[2])
            if xy_range[3] is not None:
                y_filter = y_filter | (data.y > xy_range[3])

            remove_xy = x_filter | y_filter
            if filter_type == 'any':
                remove_index = remove_index | remove_xy
            else:
                remove_index = remove_index & remove_xy

        if frame_range is not None:
            remove_frame = [False * data.shape[0]]

            if frame_range[0] is not None:
                remove_frame = remove_frame | (data.frame < frame_range[0])
            if frame_range[1] is not None:
                remove_frame = remove_frame | (data.frame > frame_range[1])

            if filter_type == 'any':
                remove_index = remove_index | remove_frame
            else:
                remove_index = remove_index & remove_frame

        return remove_index

    def _ComputerDistance(self, groundtruth, track, mode='IOU', criterion=['union']):

        if mode == 'IOU':
            return self._ComputerDistanceByIOU(groundtruth, track, criterion[0])
        elif mode == 'MinPoint':
            return self._ComputerDistanceByMinPoint(groundtruth, track)
        else:
            return None

    def _ComputeDistanceByIOU_single(self, groundtruth, track, criterion='union'):

        gt_vertex_pot = ComputeVertexCoor(groundtruth)
        tr_vertex_pot = ComputeVertexCoor(track)

        cross_pots, inner_area = RecCrossRec(gt_vertex_pot[:, :, 0], tr_vertex_pot[:, :, 0])

        gt_area = groundtruth.width * groundtruth.length
        tr_area = track.width * track.length

        if criterion == 'union':
            iou = inner_area / (gt_area + tr_area - inner_area)
        else:
            iou = inner_area / gt_area

        return iou

    def _ComputerDistanceByIOU(self, groundtruth, track, criterion='union'):

        gt_vertex_pot = ComputeVertexCoor(groundtruth)
        tr_vertex_pot = ComputeVertexCoor(track)

        gt_xy = groundtruth.loc[:, ['x', 'y']].values
        tr_xy = track.loc[:, ['x', 'y']].values

        gt_min_dis = np.sqrt(np.sum(np.square(groundtruth.loc[:, ['width', 'length']].values), axis=1)).reshape([-1, 1])
        tr_min_dis = np.sqrt(np.sum(np.square(track.loc[:, ['width', 'length']].values), axis=1)).reshape([1, -1])
        # gt_tr_min_dis = np.maximum(np.repeat(gt_min_dis, tr_min_dis.shape[1], axis = 1), np.repeat(tr_min_dis, gt_min_dis.shape[0], axis = 0))
        gt_tr_min_dis = (np.repeat(gt_min_dis, tr_min_dis.shape[1], axis=1) + np.repeat(tr_min_dis, gt_min_dis.shape[0],
                                                                                        axis=0)) * 0.5

        gt_tr_dis = ComputeDisByPots(gt_xy, tr_xy)

        iou = np.zeros([gt_vertex_pot.shape[2], tr_vertex_pot.shape[2]])

        plot = False

        for i in range(groundtruth.shape[0]):
            for j in range(track.shape[0]):
                if gt_tr_dis[i, j] > gt_tr_min_dis[i, j]:
                    continue
                else:
                    gt_index = i
                    tr_index = j

                    cross_pots, inner_area = RecCrossRec(gt_vertex_pot[:, :, gt_index], tr_vertex_pot[:, :, tr_index])

                    if plot:
                        fig = plt.figure()
                        PlotRec(gt_vertex_pot[:, :, gt_index], 'r')
                        PlotRec(tr_vertex_pot[:, :, tr_index], 'b')
                        PlotRec(cross_pots, 'k', True)

                    gt_area = groundtruth.width[gt_index] * groundtruth.length[gt_index]
                    tr_area = track.width[tr_index] * track.length[tr_index]

                    if criterion == 'union':
                        iou[gt_index, tr_index] = inner_area / (gt_area + tr_area - inner_area)
                    else:
                        iou[gt_index, tr_index] = inner_area / gt_area

                    if plot:
                        title = "frame: %d, gt: %d, tr: %d, iou: %.3f" % (
                        groundtruth.frame[gt_index], gt_index, tr_index, iou[gt_index, tr_index])
                        plt.title(title)
                        plt.savefig("/home/lz/Dataset/LidarDetection/" + title + ".png")

        return iou

    def _ComputerDistanceByMinPoint(self, groundtruth, track):

        gt_vertex_pot = ComputeVertexCoor(groundtruth)
        tr_vertex_pot = ComputeVertexCoor(track)

        gt_min_pot = ComputeMinPointInBBox(gt_vertex_pot)
        tr_min_pot = ComputeMinPointInBBox(tr_vertex_pot)

        min_pot_dis = ComputeDisByPots(gt_min_pot, tr_min_pot)

        return min_pot_dis

    def _AssociateData(self, groundtruth, track, mode='IOU', criterion=['union', 0.5], recheck=False):

        hm = Munkres()

        dis = self._ComputerDistance(groundtruth, track, mode, criterion)

        gt_ignore = groundtruth.ignore == 1

        if mode == 'IOU':
            dis = 1 - dis
            # dis[dis > (1 - self.min_IOU_Thresh)] = 1e9
            # _, tr_ignore_valid = np.where((dis <= (1 - self.min_IOU_Thresh))[gt_ignore])
            dis[dis > (1 - criterion[1])] = 1e9
            _, tr_ignore_valid = np.where((dis <= (1 - criterion[1]))[gt_ignore])
        elif mode == 'MinPoint':
            # dis[dis > self.max_MinPoint_Thresh] = 1e9
            # _, tr_ignore_valid = np.where((dis <= self.max_MinPoint_Thresh)[gt_ignore])
            dis[dis > criterion[0]] = 1e9
            _, = np.where((dis <= criterion[0])[gt_ignore])
        else:
            print("Error! Wrong mode!")
            return None, None, None

        gt_cluster_num, tr_cluster_num = dis.shape
        # association_matrix: first col is gt index and second col col is tr index
        if gt_cluster_num <= tr_cluster_num:
            association_matrix = hm.compute(dis.copy())
            association_matrix = np.array(association_matrix)
        else:
            association_matrix = hm.compute(dis.copy().T)
            association_matrix = np.array(association_matrix)[:, [1, 0]]

        dis_asso = dis[association_matrix[:, 0], association_matrix[:, 1]]
        valid_index = dis_asso < 1e9

        if mode == 'IOU':
            match_index = dis_asso <= (1 - criterion[1])
        elif mode == 'MinPoint':
            match_index = dis_asso <= criterion[0]

        if recheck:
            if mode == 'IOU':
                check_index = (dis_asso > (1 - criterion[1])) & (dis_asso <= (1 - self.min_IOU_Thresh))
            elif mode == 'MinPoint':
                check_index = (dis_asso > criterion[0]) & (dis_asso <= self.max_MinPoint_Thresh)

            if np.sum(check_index) != 0:
                gt_index = association_matrix[check_index, 0]
                tr_index = association_matrix[check_index, 1]

                gt_recheck = groundtruth.loc[gt_index, :].reset_index(drop=True)
                tr_recheck = track.loc[tr_index, :].reset_index(drop=True)

                if mode == 'IOU':
                    dis_recheck = self._ComputerDistanceByMinPoint(gt_recheck, tr_recheck)
                    recheck_true_index = np.diagonal(dis_recheck) <= self.default_MinPoint_Thresh
                    match_index[np.where(check_index)[0][recheck_true_index]] = True
                elif mode == 'MinPoint':
                    dis_recheck = self._ComputerDistanceByIOU(gt_recheck, tr_recheck)
                    recheck_true_index = np.diagonal(dis_recheck) >= self.default_IOU_Thresh
                    match_index[np.where(check_index)[0][recheck_true_index]] = True

        tr_ignore_index = np.array([False] * track.shape[0])
        tr_ignore_index[tr_ignore_valid] = True
        tr_ignore_index[association_matrix[match_index, 1]] = False

        assert np.sum(valid_index == match_index) == match_index.shape

        return association_matrix, dis, valid_index, match_index, tr_ignore_index

    def _Evaluate_Get_TP_FN_FP(self, gt, tr, mode='IOU', criterion=['union', 0.5], recheck=False):

        # gt = gt_eva.copy()
        # tr = tr_eva.copy()

        gt_tp, gt_fn, tr_tp, tr_fp, aos = 0, 0, 0, 0, 0

        empty_tr = tr.shape[0] == 0
        empty_gt = gt.shape[0] == 0
        empty_gt_nig = np.sum(gt.ignore == 0) == 0

        if empty_tr:
            if empty_gt_nig:
                return gt_tp, gt_fn, tr_tp, tr_fp, aos
            else:
                gt_fn = empty_gt_nig
                return gt_tp, gt_fn, tr_tp, tr_fp, aos
        else:
            if empty_gt:
                tr_fp = tr.shape[0]
                return gt_tp, gt_fn, tr_tp, tr_fp, aos

        association_matrix, dis, valid_index, valid_tracked_index, tr_ingore_index = self._AssociateData(gt, tr, mode,
                                                                                                         criterion,
                                                                                                         recheck)

        # valid_tracked_index = dis[association_matrix[:, 0], association_matrix[:, 1]] < 1e9

        tr.loc[:, 'ignore'] = 0
        gt_association_ignore_index = (gt.ignore[association_matrix[:, 0]] == 1).values
        tr.loc[association_matrix[gt_association_ignore_index & valid_index, 1], 'ignore'] = 1
        # tr.loc[association_matrix[gt_association_ignore_index & valid_tracked_index, 1], 'ignore'] = 1
        tr.loc[tr_ingore_index, 'ignore'] = 1

        gt['tp'] = 0
        gt['fn'] = 1
        tr['tp'] = 0
        tr['fp'] = 1

        # matched and good matched: tp
        gt.loc[association_matrix[valid_tracked_index, 0], 'tp'] = 1
        # no match or bad match in gt: fn
        # gt.loc[association_matrix[valid_index, 0], 'fn'] = 0
        gt.loc[association_matrix[valid_tracked_index, 0], 'fn'] = 0
        # matched and good matched: tp
        tr.loc[association_matrix[valid_tracked_index, 1], 'tp'] = 1
        # no match or bad match in tr: fp
        tr.loc[association_matrix[valid_tracked_index, 1], 'fp'] = 0
        # aos
        noignore_valid_tracked_index = (gt.ignore[association_matrix[:, 0]] == 0).values & valid_tracked_index
        delta = gt.heading[association_matrix[noignore_valid_tracked_index, 0]].values - tr.heading[
            association_matrix[noignore_valid_tracked_index, 1]].values

        # no ignore tp number in gt
        gt_tp = np.sum(gt.tp.values * (1 - gt.ignore.values))
        # no ignore fn number in gt
        gt_fn = np.sum(gt.fn.values * (1 - gt.ignore.values))
        # no ignore tp number in tr
        tr_tp = np.sum(tr.tp.values * (1 - tr.ignore.values))
        # no ignore fp number in tr
        tr_fp = np.sum(tr.fp.values * (1 - tr.ignore.values))
        # aos
        aos = 0 if gt_tp == 0 else np.sum(0.5 + 0.5 * np.cos(delta))

        return gt_tp, gt_fn, tr_tp, tr_fp, aos

    def _EvaluateDetection(self, groundtruth, track, detection_type=None, mode='IOU', criterion=['union', 0.5],
                           recheck=False,
                           filter_type='any', ignore_type=None, ignore_occlusion=None, ignore_truncation=None,
                           ignore_radius=None, ignore_xy_range=None):

        gt = groundtruth.copy()
        tr = track.copy()

        frame_list = sorted(groundtruth.frame.value_counts().index.tolist())

        if detection_type is None: detection_type = self.obj_type

        recall_rate = pd.DataFrame(np.zeros([len(frame_list), len(detection_type)]), columns=detection_type)
        presicion_rate = pd.DataFrame(np.zeros([len(frame_list), len(detection_type)]), columns=detection_type)
        fp_rate = pd.DataFrame(np.zeros([len(frame_list), len(detection_type)]), columns=detection_type)
        ap_rate = pd.DataFrame(np.zeros([len(frame_list), len(detection_type)]), columns=detection_type)
        aos = pd.DataFrame(np.zeros([len(frame_list), len(detection_type)]), columns=detection_type)

        # get ingore groundtruth
        gt['ignore'] = 0
        if ignore_type is not None:
            keep_obj_type = [i for i in self.obj_type if i not in ignore_type]
        else:
            keep_obj_type = self.obj_type

        ignore_index = self._filterData(gt,
                                        obj_type=keep_obj_type, occulsion=ignore_occlusion,
                                        truncation=ignore_truncation, radius=ignore_radius,
                                        xy_range=ignore_xy_range, filter_type=filter_type)
        gt.loc[ignore_index, 'ignore'] = 1

        if ignore_type is not None:
            gt_ignore_type_index = gt.apply(lambda x: x.obj_type in ignore_type, axis=1)
        else:
            gt_ignore_type_index = np.array([False] * gt.shape[0])

        for i in range(len(frame_list)):
            frame = frame_list[i]
            print(frame)

            gt_frame = gt.loc[gt.frame == frame, :].reset_index(drop=True)
            tr_frame = tr.loc[tr.frame == frame, :].reset_index(drop=True)

            for j in range(len(detection_type)):

                gt_frame_obj = gt_frame.loc[(gt_frame.obj_type == detection_type[j]).values | gt_ignore_type_index[
                    gt.frame == frame].values, :].reset_index(drop=True)
                tr_frame_obj = tr_frame.loc[tr_frame.obj_type == detection_type[j], :].reset_index(drop=True)

                gt_tp, gt_fn, tr_tp, tr_fp, aos_tp = self._Evaluate_Get_TP_FN_FP(gt_frame_obj, tr_frame_obj, mode,
                                                                                 criterion, recheck)
                # if j == 2:
                # print(gt_tp, gt_fn, tr_tp, tr_fp, aos_tp)

                assert gt_tp == tr_tp

                if (gt_fn + gt_tp) == 0:  # 可能gt总数不为0但是存在ignore
                    if (tr_tp + tr_fp) == 0:
                        recall_rate.at[i, detection_type[j]] = np.nan
                        presicion_rate.at[i, detection_type[j]] = np.nan
                        ap_rate.at[i, detection_type[j]] = np.nan
                        aos.at[i, detection_type[j]] = np.nan
                        fp_rate.at[i, detection_type[j]] = np.nan
                        continue
                    else:
                        recall_rate.at[i, detection_type[j]] = np.nan
                        presicion_rate.at[i, detection_type[j]] = tr_tp / (tr_tp + tr_fp)
                        ap_rate.at[i, detection_type[j]] = np.nan
                        aos.at[i, detection_type[j]] = np.nan
                        fp_rate.at[i, detection_type[j]] = tr_fp / (tr_tp + tr_fp)
                        continue
                else:
                    if (tr_tp + tr_fp) == 0:
                        recall_rate.at[i, detection_type[j]] = gt_tp / (gt_tp + gt_fn)
                        presicion_rate.at[i, detection_type[j]] = np.nan
                        ap_rate.at[i, detection_type[j]] = np.nan
                        fp_rate.at[i, detection_type[j]] = np.nan
                        # aos.at[i, detection_type[j]] = 0 if gt_tp == 0 else aos_tp/gt_tp
                        aos.at[i, detection_type[j]] = 0 if gt_tp == 0 else aos_tp / (gt_tp + gt_fn)
                        continue
                    else:
                        # get recal and precision
                        recall_rate.at[i, detection_type[j]] = gt_tp / (gt_tp + gt_fn)
                        presicion_rate.at[i, detection_type[j]] = tr_tp / (tr_tp + tr_fp)
                        fp_rate.at[i, detection_type[j]] = tr_fp / (tr_tp + tr_fp)
                        # aos.at[i, detection_type[j]] = 0 if gt_tp == 0 else aos_tp/gt_tp
                        aos.at[i, detection_type[j]] = 0 if gt_tp == 0 else aos_tp / (gt_tp + gt_fn)

                if tr_tp == 0:
                    ap_rate.at[i, detection_type[j]] = 0
                    continue

                # get tp score and unsame threshold
                tp_score = tr_frame_obj.loc[
                    (tr_frame_obj.tp == 1).values & (tr_frame_obj.ignore == 0).values, 'score'].values
                tp_score_sortInx = np.argsort(-tp_score)
                # 降序
                tp_score = tp_score[tp_score_sortInx]
                distinct_value_indices = np.where(np.diff(tp_score))[0]
                threshold_idxs = np.r_[distinct_value_indices, tp_score.size - 1]

                # get recall and precision by different threshold
                ap_recall = []
                ap_precision = []

                for k in range(threshold_idxs.shape[0]):
                    tr_frame_obj_thresh = tr_frame_obj.loc[tr_frame_obj.score >= tp_score[threshold_idxs[k]],
                                          :].reset_index(drop=True)
                    gt_tp_t, gt_fn_t, tr_tp_t, tr_fp_t, aos_tp_t = self._Evaluate_Get_TP_FN_FP(gt_frame_obj,
                                                                                               tr_frame_obj_thresh,
                                                                                               mode, criterion, recheck)
                    assert gt_tp_t == tr_tp_t
                    ap_recall.append(gt_tp_t / (gt_tp_t + gt_fn_t))
                    ap_precision.append(tr_tp_t / (tr_tp_t + tr_fp_t))
                    # ap_precision.append(tr_tp_t / max(tr_tp_t + tr_fp_t, 1e-6))

                # ap_recall是升序的
                ap_recall = np.array(ap_recall)
                ap_precision = np.array(ap_precision)

                # get ap without interpolating
                last_ind = ap_recall.searchsorted(
                    ap_recall[-1])  # stop when full recall attained, and reverse the outputs so recall is decreasing
                sl = slice(last_ind, None, -1)
                ap_recall = np.r_[ap_recall[sl], 0]
                ap_precision = np.r_[ap_precision[sl], 1]
                ap = -np.sum(np.diff(ap_recall) * ap_precision[:-1])
                # set ap
                ap_rate.at[i, detection_type[j]] = ap

        recall_rate['frame'] = frame_list
        presicion_rate['frame'] = frame_list
        ap_rate['frame'] = frame_list
        fp_rate['frame'] = frame_list
        aos['frame'] = frame_list

        recall_rate = recall_rate[['frame'] + detection_type]
        presicion_rate = presicion_rate[['frame'] + detection_type]
        fp_rate = fp_rate[['frame'] + detection_type]
        ap_rate = ap_rate[['frame'] + detection_type]
        aos = aos[['frame'] + detection_type]

        return recall_rate, presicion_rate, fp_rate, ap_rate, aos

    def _EvaluateTrack(self, groundtruth, track, mode='IOU', criterion=['union', 0.5], recheck=False,
                       filter_type='any', ignore_type=None, ignore_occlusion=None, ignore_truncation=None,
                       ignore_radius=None, ignore_xy_range=None):

        gt = groundtruth.copy()
        tr = track.copy()

        frame_list = sorted(gt.frame.value_counts().index.tolist())

        gt['tp'] = 0
        gt['fn'] = 1
        gt['tracked_id'] = -1
        gt['tracked_dist'] = -1
        gt['tracked_index'] = -1
        gt['tracked_valid'] = 0

        tr['tp'] = 0
        tr['fp'] = 1
        tr['frame'] *= 5

        # get ingore groundtruth
        gt['ignore'] = 0
        if ignore_type is not None:
            keep_obj_type = [i for i in self.obj_type if i not in ignore_type]
        else:
            keep_obj_type = self.obj_type

        ignore_index = self._filterData(gt,
                                        obj_type=keep_obj_type, occulsion=ignore_occlusion,
                                        truncation=ignore_truncation, radius=ignore_radius,
                                        xy_range=ignore_xy_range, filter_type=filter_type)
        ignore_index = ignore_index | (gt.track_id == -1)
        gt.loc[ignore_index, 'ignore'] = 1

        tr['ignore'] = 0

        for i in range(len(frame_list)):
            frame = frame_list[i]

            gt_frame = gt.loc[gt.frame == frame, :].reset_index(drop=True)
            tr_frame = tr.loc[tr.frame == frame, :].reset_index(drop=True)
            # print(track_id_list)

            if gt_frame.shape[0] == 0 or tr_frame.shape[0] == 0:
                continue

            association_matrix, dis, valid_index, valid_tracked_index, tr_ignore_index = self._AssociateData(gt_frame,
                                                                                                             tr_frame,
                                                                                                             mode,
                                                                                                             criterion,
                                                                                                             recheck)

            # valid_tracked_index = dis[association_matrix[:, 0], association_matrix[:, 1]] < 1e9

            gt_frame_tracked_index = np.arange(gt.shape[0])[gt.frame == frame][association_matrix[valid_index, 0]]
            tr_frame_tracked_index = np.arange(tr.shape[0])[tr.frame == frame][association_matrix[valid_index, 1]]
            gt_frame_valid_tracked_index = np.arange(gt.shape[0])[gt.frame == frame][
                association_matrix[valid_tracked_index, 0]]
            tr_frame_valid_tracked_index = np.arange(tr.shape[0])[tr.frame == frame][
                association_matrix[valid_tracked_index, 1]]
            tr_frame_ignore_index = np.arange(tr.shape[0])[tr.frame == frame][tr_ignore_index]

            gt.loc[gt_frame_valid_tracked_index, 'tp'] = 1
            gt.loc[gt_frame_valid_tracked_index, 'tracked_valid'] = 1

            gt.loc[gt_frame_tracked_index, 'fn'] = 0
            # gt.loc[gt_frame_valid_tracked_index, 'fn'] = 0
            gt.loc[gt_frame_tracked_index, 'tracked_id'] = tr.loc[tr_frame_tracked_index, 'track_id'].values
            gt.loc[gt_frame_tracked_index, 'tracked_dist'] = dis[
                association_matrix[valid_index, 0], association_matrix[valid_index, 1]]
            gt.loc[gt_frame_tracked_index, 'tracked_index'] = tr_frame_tracked_index

            tr.loc[tr_frame_valid_tracked_index, 'tp'] = 1
            tr.loc[tr_frame_valid_tracked_index, 'fp'] = 0
            tr.loc[tr_frame_ignore_index, 'ignore'] = 1

        tr.loc[gt.tracked_index[ignore_index].values[gt.tracked_index[ignore_index].values != -1], 'ignore'] = 1

        # reset tp, fp, fn
        gt['ignore_tp'] = gt.ignore * gt.tp
        gt['ignore_fn'] = gt.ignore * gt.fn
        gt['tp'] = (1 - gt.ignore) * gt.tp
        gt['fn'] = (1 - gt.ignore) * gt.fn

        tr['ignore_tp'] = tr.ignore * tr.tp
        tr['ignore_fp'] = tr.ignore * tr.fp
        tr['tp'] = (1 - tr.ignore) * tr.tp
        tr['fp'] = (1 - tr.ignore) * tr.fp

        # get cost
        gt['tp_cost'] = -1
        gt.loc[gt.tp == 1, 'tp_cost'] = gt.loc[gt.tp == 1, 'tracked_dist'].values
        if mode == 'IOU':
            gt.loc[gt.tp == 1, 'tp_cost'] = 1 - gt.loc[gt.tp == 1, 'tracked_dist'].values

        # get idswitch and framents, MT/PT/ML
        IDs = 0
        Frag = 0
        MT = 0
        PT = 0
        ML = 0
        MODP_t = []

        trajectories = {}
        traj_track_id = list(set(gt.loc[gt.ignore == 0, 'track_id'].tolist()))
        for i in range(len(traj_track_id)):
            temp_track_id = traj_track_id[i]
            temp_gt = gt.loc[(gt.ignore == 0) & (gt.track_id == temp_track_id), :]

            temp_tracked_id = temp_gt.apply(lambda x: x.tracked_id if x.tracked_valid == 1 else -1, axis=1).values
            # print(temp_tracked_id)
            trajectories[temp_track_id] = temp_tracked_id

            last_id = temp_tracked_id[0]
            for j in range(1, temp_tracked_id.shape[0]):
                # if temp_tracked_id[j] != -1 and last_id != -1 and temp_tracked_id[j - 1] != -1 and temp_tracked_id[j] != last_id:
                if temp_tracked_id[j] != -1 and last_id != -1 and temp_tracked_id[j] != last_id:
                    IDs += 1
                if j < temp_tracked_id.shape[0] - 1 and temp_tracked_id[j] != -1 and last_id != -1 and temp_tracked_id[
                    j - 1] == -1:
                    Frag += 1
                if j == temp_tracked_id.shape[0] - 1 and last_id != -1 and temp_tracked_id[j] == -1:
                    Frag += 1
                if temp_tracked_id[j] != -1:
                    last_id = temp_tracked_id[j]

            # compute MODP_t
            if np.sum(temp_gt.tp) == 0:
                MODP_t.append(1)
            else:
                MODP_t.append(np.sum(temp_gt.tp_cost[temp_gt.tp == 1]) / np.sum(temp_gt.tp))

            # compute MT/PT/ML
            tracking_ratio = np.sum(temp_tracked_id != -1) / temp_tracked_id.shape[0]

            if tracking_ratio > 0.8:
                MT += 1
            elif tracking_ratio < 0.2:
                ML += 1
            else:  # 0.2 <= tracking_ratio <= 0.8
                PT += 1

        # --get evaluate result
        res = {}
        traj_num = len(traj_track_id)
        frame_num = len(frame_list)
        n_gt = np.sum(gt.ignore == 0)

        gt_tp = np.sum(gt.tp)
        gt_fn = np.sum(gt.fn)
        tr_tp = np.sum(tr.tp)
        tr_fp = np.sum(tr.fp)

        assert gt_tp == tr_tp

        total_cost = np.sum(gt.tp_cost[gt.tp == 1])

        # tp, fn, fp, frag, ids
        res['TP'] = gt_tp
        res['FN'] = gt_fn
        res['FP'] = tr_fp
        res['Frag'] = Frag
        res['IDs'] = IDs

        # MT/PT/ML
        if traj_num == 0:
            res['MT'] = 0
            res['PT'] = 0
            res['ML'] = 0
        else:
            res['MT'] = MT / traj_num
            res['PT'] = PT / traj_num
            res['ML'] = ML / traj_num

        # precision/recall etc.
        if gt_tp + gt_fn == 0 or tr_tp + tr_fp == 0:
            res['Recall'] = 0
            res['Precision'] = 0
        else:
            res['Recall'] = gt_tp / (gt_tp + gt_fn)
            res['Precision'] = tr_tp / (tr_tp + tr_fp)

        if res['Recall'] + res['Precision'] == 0:
            res['F1'] = 0
        else:
            res['F1'] = 2. * (res['Precision'] * res['Recall']) / (res['Precision'] + res['Recall'])

        if frame_num == 0:
            res['FAR'] = "n/a"
            res['FAF'] = "n/a"
        else:
            res['FAR'] = gt_tp / frame_num
            res['FAF'] = tr_fp / frame_num

        # compute CLEARMOT
        if n_gt == 0:
            res['MOTA'] = -float('inf')
            res['MODA'] = -float('inf')
            res['MOTAL'] = -float('inf')
        else:
            res['MOTA'] = 1 - (gt_fn + tr_fp + IDs) / n_gt
            res['MODA'] = 1 - (gt_fn + tr_fp) / n_gt
            if IDs == 0:
                res['MOTAL'] = 1 - (gt_fn + tr_fp + IDs) / n_gt
            else:
                res['MOTAL'] = 1 - (gt_fn + tr_fp + np.log10(IDs)) / n_gt

        if gt_tp == 0:
            res['MOTP'] = float('inf')
        else:
            res['MOTP'] = total_cost / gt_tp

        if frame_num == 0:
            res['MODP'] = "n/a"
        else:
            res['MODP'] = sum(MODP_t) / frame_num

        return res

