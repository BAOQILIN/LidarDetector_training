import os
import yaml
import utils  # (公共文件)


class DataPreprocessor(object):
    def __init__(self, params_dict, data_root, train_flag, test_flag, res_dict={}):
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
                |       |---- 数据库id_传感器名称位置_帧号.pcd
                |       |---- ...(会同时包含对应的左/主/右雷达点云数据)
                |---- Image
                |       |---- AAA_BBB_CCC0.jpg
                |       |---- AAA_BBB_CCC1.jpg
                |       |---- 数据库id_传感器名称位置_帧号.jpg
                |       |---- ...(会同时包含对应的前/后/左/右相机图片数据)
                |---- Label
                |       |---- XXX_YYY_ZZZ0.json
                |       |---- XXX_YYY_ZZZ1.json
                |       |---- AAA_BBB_CCC0.json
                |       |---- AAA_BBB_CCC1.json
                |       |---- 数据库id_传感器名称位置_帧号.json
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
            test_flag: True/False, 说明当前数据预处理过程是用于模型测试 or 模型感知
                        当用于模型测试,则应该包含标签数据; 当用于模型感知,则只需要包含原始数据即可.
            通过train_flag 和 test_flag 两个标志参数, 区分了训练过程 / 测试过程 / 感知过程.
            
            res_dict: res_dict['msg']是一个列表,列表中的每个元素均为字符串,用户通过向此列表中添加字符串,在web前端页面打印相应的消息
        """
        self.params_dict = params_dict
        self.data_root = data_root
        self.train_flag = train_flag
        self.test_flag = test_flag

        self.ori_label_dir = os.path.join(self.data_root, self.params_dict['ORI_LABEL_PATH'][0])
        self.label_template_path = os.path.join(self.ori_label_dir, 'label.json')
        if not os.path.exists(self.label_template_path):
            self.res_dict.append('Error, No label.json!')
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

    # def func0(self, arg0, arg1, ...):
    #     """示例：自定义方法0，可删"""
    #
    # def func1(self, arg0, arg1, arg2, ...):
    #     """示例：自定义方法1，可删"""
    #
    # ...
    # "自行增加类内自定义方法"

# def function0(arg0, arg1, ...):
#     """示例：自定义类外函数0，可删"""
#
# ...
# "自行增加类外自定义函数"


if __name__ == '__main__':
    cfg_file_path = r'D:\Web_LidarDetector\Pointnet\algo\timestamp\algo_config.yaml'
    params_dict = yaml.load(open(cfg_file_path, encoding='utf-8'), Loader=yaml.FullLoader)
    data_root = r'D:\HBOX_Project\LidarDataset\main_dataset\R80_UrbanRoad_20210731_120612'
    train_flag = True
    res_dict = {'msg': []}
    data_preprocessor = DataPreprocessor(params_dict, data_root, train_flag, res_dict)
    data_preprocessor.data_preprocess()
