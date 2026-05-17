import matplotlib.pyplot as plt
import numpy as np
import os
import yaml


class DataDisplayer(object):
    def __init__(self, params_dict, result_root, save_root, res_dict={}):
        self.params_dict = params_dict
        self.result_root = result_root
        self.save_root = save_root
        self.res_dict = res_dict
        # self.loss_png_path_list = []
        # self.eva_png_path_list = []
        path = os.path.join(result_root, params_dict["RESULT_PATH"][0])
        self.losses = np.load(os.path.join(path, 'losses.npy'), allow_pickle=True).item()['loss_record']
        self.eva_result = np.load(os.path.join(path, 'eva_vali.npy'), allow_pickle=True).item()['evaluate_result']

    def display_loss(self):
        """
        函数头和返回值语句不可删改
        功能：自定义绘制loss曲线,保存图片至save_root路径下
        """

        # 训练过程的损失值从self.losses中获取
        self.losses = np.array(self.losses)
        self.res_dict['msg'].append(len(self.losses))
        self.losses = np.transpose(self.losses, (1, 0))

        # 示例(可自定义绘制多张loss曲线图片,图片名称自拟,但每张图片的执行逻辑应和示例一致)
        # loss_png_path = os.path.join(self.save_root, 'losses.png')
        # plt.savefig(loss_png_path)
        # self.loss_png_path_list.append(loss_png_path)

        losses_type = self.params_dict["LOSSES_TYPE"][0]
        fig = plt.figure(figsize=(15, 5))
        for i in range(len(losses_type)):
            plt.plot(self.losses[i])

        plt.legend(losses_type)
        loss_png_path = os.path.join(self.save_root, 'losses.png')
        plt.savefig(loss_png_path)
        self.res_dict['msg'].append('Saving %s' % loss_png_path)

    def display_eva(self):
        """
        自定义绘制验证集性能曲线,保存图片至save_root路径下
        """

        # 验证测评过程中的评估指标数据从self.eva_result中获取
        self.eva_result = np.array(self.eva_result, dtype=object)
        self.eva_result = np.transpose(self.eva_result, (1, 0))

        # 示例(可自定义绘制多张验证集性能曲线图片,图片名称自拟,但每张图片的执行逻辑应和示例一致)
        # eva_png_path = os.path.join(self.save_root, 'recall.png')
        # plt.savefig(eva_png_path)
        # self.eva_png_path_list.append(eva_png_path)

        subclass_type_1 = self.params_dict["SUBCLASS_TYPE_1"][0]
        fig = plt.figure(figsize=(15, 5))
        plt.plot(self.eva_result[0])
        plt.legend(subclass_type_1)
        eva_png_path = os.path.join(self.save_root, "eva_test_" + "subclass_type_1" + ".png")
        plt.savefig(eva_png_path)
        # self.eva_png_path_list.append(eva_png_path)

        eva_test_ap = self.eva_result[1]
        eva_test_ap = [eva_test_ap[i].tolist() for i in range(len(self.eva_result[1]))]
        list_total = np.array([eva_test_ap[i] for i in range(len(self.eva_result[1]))])
        list_total = np.transpose(list_total, (1, 0))

        subclass_type_2 = self.params_dict["SUBCLASS_TYPE_2"][0]
        fig = plt.figure(figsize=(15, 5))
        for i in range(len(list_total)):
            plt.plot(list_total[i])
        plt.legend(subclass_type_2, bbox_to_anchor=(1, 1))
        eva_png_path = os.path.join(self.save_root, "eva_test_" + "subclass_type_2" + ".png")
        plt.savefig(eva_png_path)
        self.res_dict['msg'].append('Saving %s' % eva_png_path)
        # self.eva_png_path_list.append(eva_png_path)


if __name__ == '__main__':
    cfg_file_path = r'D:\Web_LidarDetector\Pointnet\algo\timestamp\algo_config.yaml'
    params_dict = yaml.load(open(cfg_file_path, encoding='utf-8'), Loader=yaml.FullLoader)
    save_root = r'D:\HBOX_Project\LidarDataset\main_dataset\R80_UrbanRoad_20210731_120612'
    result_root = r'D:\HBOX_Project\LidarDataset\main_dataset\R80_UrbanRoad_20210731_120612'
    res_dict = {'msg': []}
    displayer = DataDisplayer(params_dict, result_root, save_root, res_dict)
    displayer.display_loss()
    displayer.display_eva()