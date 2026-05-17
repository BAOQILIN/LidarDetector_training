from DataUtils.data_evaluater_base import data_evaluater_base
import utils  # (公共文件)


# class data_evaluater_base(metaclass=abc.ABCMeta):
#     def __init__(self, params_dict, result_root, train_flag, res_dict={}):
#         self.params_dict = params_dict
#         self.result_root = result_root
#         self.train_flag = train_flag
#         self.res_dict = res_dict
#         if 'msg' not in self.res_dict:
#             self.res_dict['msg'] = []
#
#         self.result_path = os.path.join(self.result_root, self.params_dict['TRAIN']['PATH']['RESULT_PATH'][0])
#         if not os.path.exists(self.result_path):
#             os.makedirs(self.result_path)
#
#         self.evaluate_results = []
#
#         if self.train_flag:
#             self.evaluate_file = 'eva_vali.npy'
#         else:
#             self.evaluate_file = 'eva_test.npy'
#
#         if os.path.exists(os.path.join(self.result_path, self.evaluate_file)):
#             self.evaluate_results = np.load(os.path.join(self.result_path, self.evaluate_file),
#                                                                 allow_pickle=True).item()['evaluate_result']
#         if self.params_dict['TRAIN']['OVERALL']['INITIAL_RESULT'][0]:
#             self.evaluate_results = []
#
#     def save(self):
#         np.save(os.path.join(self.result_path, self.evaluate_file), {'evaluate_result': self.evaluate_results})
#
#     #    @abc.abstractmethod
#     def initial(self):
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
        self.initial()



    def initial(self):
        """函数头不可更改, 后续函数体可自定义"""
        # 用户重定义如下成员变量,用于保存整个数据集的真值和预测值,原因在于测评结果是基于整个数据集
        self.preds =
        self.labels =

    def record(self, model_outputs, labels):
        """
        函数头不可改动,函数体自定义
        功能: 接受单个batch的模型输出预测值和对应的标签真值,经过相应处理后添加到self.labels和self.preds(具体形式self.initial()中可自定义)
        model_outputs: 由model_computers.py文件中 class model_computer_base.model_compute()方法得到的单个batch模型预测值
                       同时也是networks.py文件中 class Network.forward()方法的返回值
        labels: 对应batch的真值,data_dataset.py文件中class dataset.__getitem__()方法中的返回值labels
        """

    def evaluate(self):
        """
        函数头不可改动,函数体自定义
        功能: 对整个数据集的预测值self.preds和真值self.labels进行测评,得到相关指标(eg:AP、precision、recall.etc)并保存
        """

        # 测评的指标结果必须保存到self.evaluate_results列表中
        # 代码示例如下,列表中元素的具体形式和内容可以自定义
        # 测评结束后父类中save()方法会保存测评结果self.evaluate_results到本地文件eva_vali.npy/eva_test.npy.
        self.evaluate_results.append([AP, precision, recall])

#     def func0(self, arg0, arg1, ...):
#         "示例：自定义类内方法，可增删"
#
# def function0(arg0, arg1, ...):
#     "自定义类外函数，可增删"

