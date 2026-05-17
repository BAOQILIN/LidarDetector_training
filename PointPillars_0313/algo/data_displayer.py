import matplotlib.pyplot as plt
import numpy as np
import os
import yaml


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

        path = os.path.join(result_root, params_dict["RESULT_PATH"][0])
        self.losses = np.load(os.path.join(path, 'losses.npy'), allow_pickle=True).item()['loss_record']
        self.eva_result = np.load(os.path.join(path, 'eva_vali.npy'), allow_pickle=True).item()['evaluate_result']

        self.class_names = ['Pedestrian', 'Mbike', 'Car', 'Bus', 'Tricycle']

    def display_loss(self):
        """
        函数头不可删改
        功能: 自定义绘制loss曲线,保存图片至save_root路径下,图片名称自定义
        """

        # 训练过程的损失值从self.losses中获取,用户自定义处理过程
        cls_num = len(self.class_names)

        type_cls_losses = [[res[0][i] for res in self.losses] for i in range(cls_num)]
        for type_cls_loss in type_cls_losses:
            plt.plot(type_cls_loss)
        plt.legend(self.class_names)
        plt.xlabel('iterations')
        plt.ylabel('type_cls_losses')
        plt.savefig(os.path.join(self.picture_save_root, 'type_class_losses.png'), bbox_inches='tight')
        plt.close()
        self.res_dict['msg'].append('Saving %s' % os.path.join(self.picture_save_root, 'type_class_losses.png'))

        type_reg_losses = [[res[1][i] for res in self.losses] for i in range(cls_num)]
        for type_reg_loss in type_reg_losses:
            plt.plot(type_reg_loss)
        plt.legend(self.class_names)
        plt.xlabel('iteration')
        plt.ylabel('type_reg_losses')
        plt.savefig(os.path.join(self.picture_save_root, 'type_reg_losses.png'), bbox_inches='tight')
        plt.close()
        self.res_dict['msg'].append('Saving %s' % os.path.join(self.picture_save_root, 'type_reg_losses.png'))

        cls_loss = [ret[2] for ret in self.losses]
        reg_loss = [ret[3] for ret in self.losses]
        total_loss = [ret[4] for ret in self.losses]
        cls_loss_pos = [ret[5] for ret in self.losses]
        cls_loss_neg = [i - j for i, j in zip(cls_loss, cls_loss_pos)]
        plt.plot(cls_loss)
        plt.plot(reg_loss)
        plt.plot(total_loss)
        plt.legend(['cls_loss', 'reg_loss', 'total_loss'])
        plt.xlabel('iterations')
        plt.ylabel('total_loss')
        plt.savefig(os.path.join(self.picture_save_root, 'total_loss.png'), bbox_inches='tight')
        plt.close()
        self.res_dict['msg'].append('Saving %s' % os.path.join(self.picture_save_root, 'total_loss.png'))

        plt.plot(cls_loss_pos)
        plt.legend(['cls_loss_pos'])
        plt.xlabel('iterations')
        plt.ylabel('cls_loss_pos')
        plt.savefig(os.path.join(self.picture_save_root, 'cls_loss_pos.png'), bbox_inches='tight')
        plt.close()
        self.res_dict['msg'].append('Saving %s' % os.path.join(self.picture_save_root, 'cls_loss_pos.png'))

        plt.plot(cls_loss_neg)
        plt.legend(['cls_loss_neg'])
        plt.xlabel('iterations')
        plt.ylabel('cls_loss_neg')
        plt.savefig(os.path.join(self.picture_save_root, 'cls_loss_neg.png'), bbox_inches='tight')
        plt.close()
        self.res_dict['msg'].append('Saving %s' % os.path.join(self.picture_save_root, 'cls_loss_neg.png'))

        cls_loss_pos_ratio = [(i / j) for i, j in zip(cls_loss_pos, cls_loss)]
        plt.plot(cls_loss_pos_ratio)
        plt.legend(['cls_loss_pos_ratio'])
        plt.xlabel('iterations')
        plt.ylabel('cls_loss_pos_ratio')
        plt.savefig(os.path.join(self.picture_save_root, 'cls_loss_pos_ratio.png'), bbox_inches='tight')
        plt.close()
        self.res_dict['msg'].append('Saving %s' % os.path.join(self.picture_save_root, 'cls_loss_pos_ratio.png'))

        # 示例(可自定义绘制多张loss曲线图片,图片名称自拟,但每张图片的执行逻辑应和示例一致)
        # loss_png_path = os.path.join(self.picture_save_root, 'losses.png')
        # plt.savefig(loss_png_path)
        # self.res_dict['msg'].append('Saving %s' % loss_png_path)

    def display_eva(self):
        """
        函数头不可删改
        功能: 自定义绘制验证集性能曲线,保存图片至save_root路径下,图片名称自定义
        """
        cls_num = len(self.class_names)
        [recalls, precisions, aps, aoses] = [[res[i] for res in self.eva_result] for i in range(4)]

        recall_each_type = [[recall[i] for recall in recalls] for i in range(cls_num)]
        for cur_recall in recall_each_type:
            plt.plot(cur_recall)
        plt.xlabel('epoch')
        plt.ylabel('recall')
        plt.legend(self.class_names)
        plt.savefig(os.path.join(self.picture_save_root, 'recalls.png'), bbox_inches='tight')
        self.res_dict['msg'].append('Saving %s' % os.path.join(self.picture_save_root, 'recalls.png'))

        precision_each_type = [[precision[i] for precision in precisions] for i in range(cls_num)]
        for cur_precision in precision_each_type:
            plt.plot(cur_precision)
        plt.xlabel('epoch')
        plt.ylabel('precision')
        plt.legend(self.class_names)
        plt.savefig(os.path.join(self.picture_save_root, 'precisions.png'), bbox_inches='tight')
        self.res_dict['msg'].append('Saving %s' % os.path.join(self.picture_save_root, 'precisions.png'))

        ap_each_type = [[ap[i] for ap in aps] for i in range(cls_num)]
        for cur_ap in ap_each_type:
            plt.plot(cur_ap)
        plt.xlabel('epoch')
        plt.ylabel('ap')
        plt.legend(self.class_names)
        plt.savefig(os.path.join(self.picture_save_root, 'aps.png'), bbox_inches='tight')
        self.res_dict['msg'].append('Saving %s' % os.path.join(self.picture_save_root, 'aps.png'))

        aos_each_type = [[aos[i] for aos in aoses] for i in range(cls_num)]
        for cur_aos in aos_each_type:
            plt.plot(cur_aos)
        plt.xlabel('epoch')
        plt.ylabel('aos')
        plt.legend(self.class_names)
        plt.savefig(os.path.join(self.picture_save_root, 'aoses.png'), bbox_inches='tight')
        self.res_dict['msg'].append('Saving %s' % os.path.join(self.picture_save_root, 'aoses.png'))

        # 验证测评过程中的评估指标数据从self.eva_result中获取,用户自定义处理过程
        # 示例如下:
        # self.eva_result = np.array(self.eva_result, dtype=object)
        # self.eva_result = np.transpose(self.eva_result, (1, 0))

        # 示例(可自定义绘制多张验证集性能曲线图片,图片名称自拟,但每张图片的执行逻辑应和示例一致)
        # eva_png_path = os.path.join(self.picture_save_root, 'recall.png')
        # plt.savefig(eva_png_path)
        # self.res_dict['msg'].append('Saving %s' % loss_png_path)

