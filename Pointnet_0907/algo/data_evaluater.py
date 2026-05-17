import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from DataUtils.data_evaluater_base import data_evaluater_base


# class data_evaluater_base(metaclass=abc.ABCMeta):
#     """父类中内容供参考,请勿增删改"""
#     def __init__(self, params_dict, result_root, res_dict={}):
#         self.params_dict = params_dict
#         self.result_root = result_root
#         self.res_dict = res_dict
#         if 'msg' not in self.res_dict:
#             self.res_dict['msg'] = []
#
#         result_path = os.path.join(self.result_root, self.params_dict['TRAIN']['PATH']['RESULT_PATH'][0])
#         if not os.path.exists(result_path):
#             os.makedirs(result_path)
#
#         self.evaluate_results = {}
#         for i in range(len(self.params_dict['TEST']['TEST_']['TYPE'][0])):
#             evaluate_type = self.params_dict['TEST']['TEST_']['TYPE'][0][i]
#             evaluate_file = self.params_dict['TEST']['TEST_']['TYPE'][0][i] + '.npy'
#             if not os.path.exists(os.path.join(result_path, evaluate_file)):
#                 self.evaluate_results[evaluate_type] = {}
#             else:
#                 self.evaluate_results[evaluate_type] = np.load(os.path.join(result_path, evaluate_file),
#                                                                allow_pickle=True).item()
#             if self.params_dict['TRAIN']['OVERALL']['INITIAL_RESULT'][0]:
#                 self.evaluate_results[evaluate_type] = []
#
#     def save(self, dataset_type='eva_train'):
#         for i in range(len(self.params_dict['TEST']['TEST_']['TYPE'][0])):
#             if dataset_type == self.params_dict['TEST']['TEST_']['TYPE'][0][i]:
#                 np.save(os.path.join(self.result_root, self.params_dict['TRAIN']['PATH']['RESULT_PATH'][0],
#                                      self.params_dict['TEST']['TEST_']['TYPE'][0][i] + '.npy'),
#                         self.evaluate_results[dataset_type])
#
#     #    @abc.abstractmethod
#     def initial(self, dataset_type='eva_train'):
#         pass
#
#     #    @abc.abstractmethod
#     def record(self, model_outputs, true):
#         pass
#
#     #    @abc.abstractmethod
#     def evaluate(self):
#         pass


class data_evaluater(data_evaluater_base):
    def __init__(self, params_dict, result_root, train_flag, res_dict={}):
        super().__init__(params_dict, result_root, train_flag,  res_dict=res_dict)

        # 可以初始化如下成员变量,用于保存整个数据集的真值和预测值,可自定义
        self.preds = [[], [], []]
        self.true = [[], [], []]

    def initial(self):
        self.preds = [[], [], []]
        self.true = [[], [], []]

    def record(self, model_outputs, true):
        """
        函数头不可增删改,函数体自定义
        功能: 接受单个batch的模型输出预测值和对应的标签真值,经过相应处理后添加到self.true和self.preds(具体形式可自定义)
        model_outputs: 模型输出的单个batch的预测值
        true: 对应batch的真值
        """
        mask = true[-1:]
        self.preds[0].extend(list(model_outputs[0][mask[0]].view(-1).detach().cpu().numpy()))
        self.true[0].extend(list(true[0][mask[0]].detach().cpu().numpy()))

        heading_dic = model_outputs[1][mask[0]].detach().cpu().numpy()
        self.preds[1].extend(list(np.exp(heading_dic - np.max(heading_dic, axis=1, keepdims=True))))
        self.true[1].extend(list(true[1][mask[0]].detach().cpu().numpy()))

        type_class = model_outputs[2].detach().cpu().numpy()
        self.preds[2].extend(list(np.exp(type_class - np.max(type_class, axis=1, keepdims=True))))
        self.true[2].extend(list(true[2].detach().cpu().numpy()))

    def evaluate(self):
        """
        函数体自定义
        功能：依据相关指标(eg:AP、precision、recall.etc),对整个数据集的预测值和标签进行测评
        """
        for i in range(3):
            self.preds[i] = np.array(self.preds[i])
            self.true[i] = np.array(self.true[i])

        post_heading_error = self._evaluate_heading(self.preds[0], np.argmax(self.preds[1], axis=1), self.true[0],
                                                    self.true[1],
                                                    self.params_dict['TRAIN']['MODEL']['STRUCTURE']['HEADING_BIN'][0])
        post_heading_error = np.mean(post_heading_error)
        mAP = self._evaluate_class(self.preds[2], self.true[2])
        mAP_mean = np.mean(mAP[~np.isnan(mAP)])

        self.res_dict['msg'].append("heading error: %.6f" % (post_heading_error))
        self.res_dict['msg'].append("AP of classification: ")
        self.res_dict['msg'].append(mAP)
        self.res_dict['msg'].append("mAP of classification: %.6f" % (mAP_mean))

        # 测评的指标结果必须保存到如下的列表中,父类中save()方法会保存测评结果到本地
        self.evaluate_results.append([post_heading_error, mAP])

    def _evaluate_heading(self, pred_heading, pred_direction, true_heading, true_direction, heading_bin=4):
        """
        Args:
            pred_heading: numpy (N,)
            pred_direction: numpy (N,)
            true_heading: numpy (N,)
            true_direction: numpy (N,)
            heading_bin: default 4
        Returns:
            post_heading_error
        """
        post_pred_heading = self.data_postprocess_heading(pred_heading, pred_direction, heading_bin)
        post_true_heading = self.data_postprocess_heading(true_heading, true_direction, heading_bin)

        post_heading_error = self.data_postprocess_heading_error(post_pred_heading, post_true_heading)

        return post_heading_error

    @staticmethod
    def data_postprocess_heading(heading, direction, binNum=4):
        heading_array = np.array(heading)
        direction_array = np.array(direction)
        if binNum < 1:
            return heading_array

        binStep = 2.0 * np.pi / binNum
        for i in range(binNum):
            start_angle = binStep * i
            end_angle = binStep * (i + 1)
            index = direction_array == i
            if i % 2 == 0:
                heading_array[index] = heading_array[index] + start_angle
            else:
                heading_array[index] = end_angle - heading_array[index]
        heading_array[heading_array > np.pi] -= 2.0 * np.pi

        return heading_array

    @staticmethod
    def data_postprocess_heading_error(pred, real):
        error1 = np.abs(pred - real)
        error2 = np.abs(pred - real + np.pi)
        error3 = np.abs(pred - real - np.pi)

        error = np.array([error1, error2, error3])
        error = np.min(error, axis=0)

        error = np.array([error, np.abs(error - np.pi)])
        error = np.min(error, axis=0)

        return error

    def _evaluate_class(self, class_score, true_class):
        """
        Args:
            class_score: numpy, (N, M), M is label number, score, do not do softmax
            true_class: numpy, (N,), true class label
        Returns:
            mAP: ap of each class
        """
        label_number = class_score.shape[1]

        pred_prob = class_score / np.sum(class_score, axis=1, keepdims=True)

        label = label_binarize(true_class, np.arange(label_number))

        mAP = average_precision_score(label, pred_prob, average=None)

        return mAP


#     def func0(self, arg0, arg1, ...):
#         "示例：自定义类内方法，可增删"
#
#
# def function0(arg0, arg1, ...):
#     "自定义类外函数，可增删"

