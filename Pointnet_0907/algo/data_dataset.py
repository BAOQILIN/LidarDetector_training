import os
import math
import numpy as np
import torch
from torch.autograd import Variable
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
                或者params_dict['TEST']['TEST_']['PREFIX'][0]
                根据前缀来明确当前需要生成的是 训练阶段 或者 验证阶段 或者 测试阶段 的数据集,不同阶段应该读取不同的预处理后(data_preprocess.py)的文件.
        res_dict: res_dict['msg']是一个列表,列表中的每个元素均为字符串,用户通过向此列表中添加字符串,在web前端页面打印相应的消息
        """
        super(dataset, self).__init__()
        self.params_dict = params_dict
        self.folder = folder
        self.prefix = prefix
        self.res_dict = res_dict

        self.points = np.load(os.path.join(folder, prefix + '_points.npy'))
        self.heading = np.load(os.path.join(folder, prefix + '_headings.npy'))
        self.label = np.load(os.path.join(folder, prefix + '_label.npy'))

        f = open(os.path.join(folder, prefix + '_files.txt'), 'r')
        self.file_list = np.array(f.readlines())
        f.close()

        self.total_index = np.arange(self.points.shape[0])
        self.heading_index = self.total_index[self.heading != -1000]
        self.unheading_index = self.total_index[self.heading == -1000]

        if self.params_dict['TRAIN']['CTRL']['DATA']['KEEP_POSITIVE_DATA_RATIO'][0]:
            self.heading_num = min(self.heading_index.shape[0], int(
                                   self.params_dict['TRAIN']['CTRL']['DATA']['BATCH_SIZE'][0] *
                                   self.params_dict['TRAIN']['CTRL']['DATA']['POSITIVE_DATA_RATIO'][0]),
                                    self.points.shape[0])
            self.unheading_num = min(self.unheading_index.shape[0],
                                     self.params_dict['TRAIN']['CTRL']['DATA']['BATCH_SIZE'][0] - self.heading_num,
                                     self.points.shape[0])
            if self.heading_num == 0:
                self.num_batch = math.ceil(self.unheading_index.shape[0] / self.unheading_num)
            elif self.unheading_num == 0:
                self.num_batch = math.ceil(self.heading_index.shape[0] / self.heading_num)
            else:
                self.num_batch = max(math.ceil(self.unheading_index.shape[0] / self.unheading_num),
                                     math.ceil(self.heading_index.shape[0] / self.heading_num))
        else:
            self.num_batch = math.ceil(
                self.total_index.shape[0] / self.params_dict['TRAIN']['CTRL']['DATA']['BATCH_SIZE'][0])

        # 单个epoch中batch的个数,必须实现
        # self.num_batch =
        # 单个epoch中样本的个数,必须实现
        self.num_samples = self.points.shape[0]

    def __len__(self):
        """函数头不可删改,返回数据集中样本的个数,必须完善返回值"""
        return self.num_samples

    def __getitem__(self, index):
        """
        函数头不可删改
        功能：根据index返回训练集/验证集中单个batch的所有内容,由data_loader.py中data_loader_XXX.__next__()方法中调用
            可能需要判断index和self.num_batch之间的大小关系,从而决定何时结束迭代
        """
        if index > self.num_batch:
            return None, None, None

        if self.params_dict['TRAIN']['CTRL']['DATA']['KEEP_POSITIVE_DATA_RATIO'][0]:
            heading_index = np.random.choice(self.heading_index, self.heading_num)
            unheading_index = np.random.choice(self.unheading_index, self.unheading_num)
            batch_index = np.r_[heading_index, unheading_index]
        else:
            batch_index = np.random.choice(self.total_index, self.params_dict['TRAIN']['CTRL']['DATA']['BATCH_SIZE'][0])

        heading = self.heading[batch_index]
        heading_mask = heading != -1000
        heading_reg, heading_cls = self.data_preprocess_heading(heading, self.params_dict['TRAIN']['MODEL']['STRUCTURE']['HEADING_BIN'][0])
        heading_reg[~heading_mask] = -1000

        points = torch.FloatTensor(self.points[batch_index, :, :self.params_dict['TRAIN']['MODEL']['STRUCTURE']['INPUT_DIM'][0]])
        heading_reg = torch.FloatTensor(heading_reg)
        heading_cls = torch.LongTensor(heading_cls)
        label = torch.LongTensor(self.label[batch_index])

        if points is None or heading_reg is None or heading_cls is None or label is None:
            self.res_dict['msg'].append('dataset[%d] return value is None!!!' % index)
            raise ValueError

        heading_mask = heading_reg != -1000

        # 转换成GPU计算,示例如下:
        if torch.cuda.is_available():
            points = Variable(points.cuda())
            heading_reg = Variable(heading_reg.cuda())
            heading_cls = Variable(heading_cls.cuda())
            label = Variable(label.cuda())
        else:
            self.res_dict['msg'].append('torch.cuda.is_availabel()=False!!!')
            raise Exception

        inputs = [points]
        labels = [heading_reg, heading_cls, label, heading_mask]
        filenames = [None]

        # 返回内容的接口必须固定且必须实现
        # inputs直接被传参到 model_computer.py文件中model_computer.model_compute()方法进行前向传播计算
        # labels直接传参到loss_computers.py文件中loss_computer.loss_compute()方法中进行损失计算
        # filenames: batch_size个样本的文件名,以列表的形式组织,len(filenames)=batch_size,用于以同名文件的方式保存样本的感知结果
        return inputs, labels, filenames

    @staticmethod
    def data_preprocess_heading(heading, binNum=4):
        heading_array = np.array(heading)
        if binNum < 1:
            return heading_array, None
        # 0 <= heading < 2.0*np.pi
        heading_array[heading_array < 0] += 2.0 * np.pi

        binStep = 2.0 * np.pi / binNum
        direction = np.zeros_like(heading_array)

        for i in range(binNum):
            start_angle = binStep * i
            end_angle = binStep * (i + 1)
            index = (heading_array >= start_angle) & (heading_array < end_angle)
            if i % 2 == 0:
                heading_array[index] = heading_array[index] - start_angle
            else:
                heading_array[index] = end_angle - heading_array[index]
            direction[index] = i

        return heading_array, direction

    def __set_params(self, params_dict):
        for k, v in params_dict.items():
            exec('self.' + k.lower() + '=v')



