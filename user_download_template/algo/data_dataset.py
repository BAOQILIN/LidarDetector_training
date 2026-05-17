import utils  # (公共文件)
from torch.utils import data


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

        # # 用户需要根据self.prefix来判断当前处于训练过程 or 测试过程,可参考如下示例
        # if self.prefix == self.params_dict['TRAIN']['OVERALL']['TRAIN_PREFIX'][0]: # 说明当前处于训练状态,需要加载训练集文件,生成训练过程的dataset的实例化对象
        #     # 加载训练集文件并做相关处理
        #
        # elif self.prefix == self.params_dict['VALIDATION']['VALI_']['PREFIX'][0]: # 说明当前处于验证测试状态,需要加载验证集文件,生成验证测试过程的dataset的实例化对象
        #     # 加载验证集文件并做相关处理
        # elif self.prefix == self.params_dict['TEST']['TEST_']['PREFIX'][0]::
              # 加载测试集文件并做相关处理
        # else:
        #     # 加载模型感知数据并做相关处理

        # """ ... 可自定义添加函数体内容 ... """

        # 单个epoch中batch的个数,必须实现
        # self.num_batch =
        # 单个epoch中样本的个数,必须实现
        # self.num_samples =

    def __len__(self):
        """函数头不可删改,返回数据集中样本的个数,必须完善返回值"""
        return self.num_samples

    def __getitem__(self, index):
        """
        函数头不可删改
        功能: 根据index返回训练集/验证集/测试集/感知推理数据集 中单个batch的所有内容
              返回给loss_computers.py文件loss_computer.loss_compute()方法 和 data_evaluater.py文件data_evaluater.record()方法
              需要判断index和self.num_batch之间的大小关系,从而决定何时结束迭代
        """
        if index > self.num_batch:
            return None, None, None

        # 函数返回值inputs和labels需要转换成GPU计算
        # inputs等返回值的数据类型并不确定,但应该是由torch.gpu_tensor组成,以方便进行前向传播和损失计算,示例如下
        # if torch.cuda.is_available():
        #     inputs = Variable(inputs.cuda())
        #     labels = Variable(labels.cuda())
        # else:
        #     self.res_dict['msg'].append('torch.cuda.is_availabel()=False!!!')
        #     raise Exceptions

        # 返回内容的接口必须固定且必须实现
        # inputs直接被传参到 networks.py文件中class Network.forward()方法进行前向传播计算
        # labels直接传参到loss_computers.py文件中loss_computer.loss_compute()方法中进行损失计算
        # labels还会传参到data_evaluater.py文件中data_evaluater.record()方法中进行性能测评计算
        # filenames: batch_size个样本的文件名,以列表的形式组织,len(filenames)=batch_size,用于以同名文件的方式保存每个样本的感知结果
        # filenames传参到data_postprocessor.py文件中的data_postprocess()函数中
        return inputs, labels, filenames



