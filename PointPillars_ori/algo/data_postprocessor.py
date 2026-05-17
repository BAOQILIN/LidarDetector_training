import json
import numpy as np
import copy
import utils


class data_postprocessor(object):
    def __init__(self, label_template_path, params_dict):
        """
            此类实现 模型预测输出 --> 指定的JSON数据格式(同数据管理平台单帧标签格式)
            构造函数内容不可删改,但可以自定义增加
            label_template_path: 标签模板label.json所对应的路径,用户可自行解析获取内容
            params_dict: algo_config.yaml文件中 TRAIN_MODEL模块对应的参数字典
            """
        self.label_template_path = label_template_path
        self.params_dict = params_dict

        self.class_names = np.array(['Pedestrian', 'Mbike', 'Car', 'Bus', 'Tricycle'])
        # self.id_2_name_map = {26: 'Pedestrian', 39: 'Mbike', 157: 'Car', 156: 'Bus', 229: 'Tricycle',
        #                       230: 'Pedestrian', 227: 'Mbike'}
        # self.name_2_id_map = {'Pedestrian': 26, 'Mbike': 39, 'Car': 157, 'Bus': 156, 'Tricycle': 229}

        self.id_2_name_map = {}
        with open(self.label_template_path, 'rb') as f:
            temp = json.load(f, encoding='utf-8')
        for each in temp:
            if each['task'] == '目标检测':
                label_json = each['label']
                for obj in label_json:
                    self.id_2_name_map[obj['id']] = obj['name']

        self.name_2_id_map = {}
        for k, v in self.id_2_name_map.items():
            self.name_2_id_map.update({v: k})

    def data_postprocess(self, model_outputs, filenames):
        """
        函数头不可更改，函数体自定义
        功能: 接收单个batch的模型输出预测值, 将其转化成指定的JSON数据格式. 主要用于模型感知阶段.
        model_outputs: 模型输出的单个batch的预测值, 亦即用户自定义的Networks.forward()方法的返回值.
        filenames: 单个batch中每帧样本的原始文件名,亦即用户自定义的data_dataset.py中Dataset. __getitem__()方法的返回值.
        return: 指定格式的预测结果,格式如下:
                predictions = [prediction of sample1, prediction of sample2 ...] 不同帧样本的结果以列表形式组织
                prediction of sample1 --> 某一帧样本的结果以JSON格式组织(亦即Python中的dict字典数据类型),
                                          可以表示为{'annotaion': [prediction of object1, prediction of object2, ...]};
                prediction of object1 --> 某一帧样本中的某一个目标的预测结果,可以表示为{'position': [x, y, z], 'dimension':[w, l , h], ...}
                                          具体每帧样本标签应包含的信息,需结合数据管理平台获取的《label.json》和《每帧标签格式说明.json》
                filenames: 函数接口传入的实参filenames
                task_type: 任务类型
                
        注意：务必保证返回的预测结果和文件名称相对应.
        """

        predictions = []
        task_type = '目标检测'
        batch_cls_preds = model_outputs['cls_preds']  # list:5 (shape=(b, 128*128*2*C', C') C'代表该检测头所对应的类别个数
        batch_box_preds = model_outputs['box_preds']  # list:5 (shape=(b, 128*128*2*C', 8) C'代表检测头所对应的的类别个数
        batch_size = batch_cls_preds[0].shape[0]

        params_dict_copy = copy.deepcopy(self.params_dict)
        bbox_generator = utils.BboxGenerator(params_dict_copy)
        pred_dicts = bbox_generator.generate_predicted_boxes(batch_size, batch_cls_preds, batch_box_preds)

        for index, box_dict in enumerate(pred_dicts):
            cur_pred = {'annotation': []}
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = self.class_names[box_dict['pred_labels'].cpu().numpy() - 1].tolist()
            pred_labels = [self.name_2_id_map[i] for i in pred_labels]

            num_preds = pred_scores.shape[0]
            if num_preds != 0:
                for i in range(num_preds):
                    anno_temp = {'type': 'cuboid'}
                    anno_temp['id'] = i + 1
                    anno_temp['track_id'] = 1000
                    anno_temp['label_id'] = pred_labels[i]
                    anno_temp['position'] = pred_boxes[i, :3].tolist()
                    anno_temp['dimension'] = pred_boxes[i, 3:6].tolist()
                    anno_temp['rotation'] = [0, 0, pred_boxes[i, 6]]
                    anno_temp['velocity'] = [0, 0, 0]
                    anno_temp['acceleration'] = [0, 0, 0]
                    anno_temp['description'] = {}
                    cur_pred['annotation'].append(anno_temp)

            predictions.append(cur_pred)

        return predictions, filenames, task_type
