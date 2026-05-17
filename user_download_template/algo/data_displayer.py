import matplotlib.pyplot as plt
import numpy as np
import os
import utils  # (公共文件)


class DataDisplayer(object):
    def __init__(self, params_dict, result_root, picture_save_root, res_dict={}):
        """
        函数头不可改动,函数体不可删改,可增加内容
        此类实现 训练损失losses.npy文件的损失曲线图片生成  验证集测评文件eva_vali.npy的测评指标结果图片生成
        :param params_dict: algo_config.yaml文件 DISPLAYER模块 对应的参数字典,由用户自定义其中的具体内容
        :param result_root: 训练损失文件losses.npy和验证集测评文件eva_vali.npy的根目录,函数体中具体描述了如何获取其中数据
                            文件结构参见《文件层级结构.txt》
        :param picture_save_root: 损失曲线图片和测评指标图片保存根目录,由系统提供,用户不可见
        :param res_dict: 字典结构,res_dict['msg']为列表,列表中的元素为字符串,前端将显示打印其中的所有元素,用户可自行向其中添加字符串
        """
        self.params_dict = params_dict
        self.result_root = result_root
        self.picture_save_root = picture_save_root
        self.res_dict = res_dict
        if 'msg' not in self.res_dict:
            self.res_dict['msg'] = []

        path = os.path.join(result_root, params_dict["DISPLAY"]["RESULT_PATH"][0])
        self.losses = np.load(os.path.join(path, 'losses.npy'), allow_pickle=True).item()['loss_record']
        self.eva_result = np.load(os.path.join(path, 'eva_vali.npy'), allow_pickle=True).item()['evaluate_result']

    def display_loss(self):
        """
        函数头不可删改
        功能: 自定义绘制loss曲线,保存图片至save_root路径下,图片名称自定义
        """

        # 训练过程的损失值从self.losses中获取,用户自定义处理过程

        # 示例(可自定义绘制多张loss曲线图片,图片名称自拟,但每张图片的执行逻辑应和示例一致)
        # loss_png_path = os.path.join(self.picture_save_root, 'losses.png')
        # plt.savefig(loss_png_path)
        # self.res_dict['msg'].append('Saving %s' % loss_png_path)

    def display_eva(self):
        """
        函数头不可删改
        功能: 自定义绘制验证集性能曲线,保存图片至save_root路径下,图片名称自定义
        """

        # 验证测评过程中的评估指标数据从self.eva_result中获取,用户自定义处理过程
        # 示例如下:
        # self.eva_result = np.array(self.eva_result, dtype=object)
        # self.eva_result = np.transpose(self.eva_result, (1, 0))

        # 示例(可自定义绘制多张验证集性能曲线图片,图片名称自拟,但每张图片的执行逻辑应和示例一致)
        # eva_png_path = os.path.join(self.picture_save_root, 'recall.png')
        # plt.savefig(eva_png_path)
        # self.res_dict['msg'].append('Saving %s' % loss_png_path)

