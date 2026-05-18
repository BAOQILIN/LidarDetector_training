import os.path
import numpy as np
import torch
import copy
from ModelUtils.loss_computers_base import loss_computer_base
import utils
import sys
sys.path.insert(0, os.path.dirname(__file__))


# class loss_computer_base(metaclass=abc.ABCMeta):
#     def __init__(self, params_dict, result_root, res_dict={}):
#         self.params_dict = params_dict
#         self.result_root = result_root
#         self.res_dict = res_dict
#         if 'msg' not in self.res_dict:
#             self.res_dict['msg'] = []

#         result_path = os.path.join(self.result_root, self.params_dict['TRAIN']['PATH']['RESULT_PATH'][0])
#         if not os.path.exists(result_path):
#             os.makedirs(result_path)

#         if not os.path.exists(os.path.join(result_path, 'losses.npy')):
#             self.loss_record = []
#         else:
#             self.loss_record = self.load()

#         if self.params_dict['TRAIN']['OVERALL']['INITIAL_RESULT'][0]:
#             self.loss_record = []

#     #    @abc.abstractmethod
#     def loss_compute(self, outputs, labels, record=False):
#         pass

#     #    @abc.abstractmethod
#     def _loss_initial(self):
#         pass

#     def load(self):
#         result_path = os.path.join(self.result_root, self.params_dict['TRAIN']['PATH']['RESULT_PATH'][0])
#         return np.load(os.path.join(result_path, 'losses.npy'), allow_pickle=True).item()['loss_record']

#     def save(self):
#         result_path = os.path.join(self.result_root, self.params_dict['TRAIN']['PATH']['RESULT_PATH'][0])
#         np.save(os.path.join(result_path, 'losses.npy'), {'loss_record': self.loss_record})

#     def initial(self):
#         self.loss_record = []


class loss_computer(loss_computer_base):
    def __init__(self, params_dict, result_root, res_dict={}):
        """
        函数头固定接头,不可增删改;函数体不可删改,可自定义增加内容
        params_dict: algo_config.yaml中 TRAIN_MODEL模块对应的参数字典
        result_root: 保存损失结果文件losses.npy的根目录,具体的保存路径见loss_computer_base.save()方法
                     关于result_root的文件夹结构,参见《文件层级结构.txt》
        res_dict: res_dict['msg']是一个列表,列表中的每个元素均为字符串,用户通过向此列表中添加字符串,在web前端页面打印相应的消息
        """
        super().__init__(params_dict, result_root, res_dict)
        self._loss_initial()

        self.class_names = ['Pedestrian', 'Mbike', 'Car', 'Bus', 'Tricycle']
        self.num_class = len(self.class_names)

        self.cls_weight = self.params_dict['TRAIN']['CTRL']['LOSS']['CLS_WEIGHT'][0]
        self.loc_weight = self.params_dict['TRAIN']['CTRL']['LOSS']['LOC_WEIGHT'][0]
        self.pos_cls_weight = self.params_dict['TRAIN']['CTRL']['LOSS']['POS_CLS_WEIGHT'][0]
        self.neg_cls_weight = self.params_dict['TRAIN']['CTRL']['LOSS']['NEG_CLS_WEIGHT'][0]
        self.head_weights = [0.3, 1.0, 1.0, 1.0, 1.0]
        self.latest_metrics = {}

        self.params_dict_copy = copy.deepcopy(self.params_dict)
        bbox_generator = utils.BboxGenerator(self.params_dict_copy)
        self.point_cloud_range = bbox_generator.roi()
        self.grid_size = bbox_generator.gridsize()
        self.box_coder = bbox_generator.boxcoder()
        self.anchors = bbox_generator.anchor()

        self.target_assigner = AxisAlignedTargetAssigner(model_cfg=self.params_dict_copy,
                                                         class_names=self.class_names,
                                                         box_coder=self.box_coder)

    def _loss_initial(self):
        """
        函数头固定接口，不可增删改;函数体自定义
        以列表形式保存每个batch中生成的loss,可以根据loss类型自定义,
        在loss_compute()方法中,可以自行对此处的损失保存到self.loss_record变量中,
        框架会自动调用loss_computer_base.save()方法保存到指定文件losses.npy中
        eg: self.box_losses = [] 保存预测框的回归损失
            self.cls_losses = [] 保存预测框的类别损失
            self.total_loss = [] 总损失
        """
        self.type_cls_losses = []
        self.type_reg_losses = []
        self.class_losses = []
        self.class_losses_pos = []
        self.box_losses = []
        self.total_losses = []

    def loss_compute(self, model_outputs, labels, record=False):
        """
        函数头固定接口,不可增删改
        功能: 计算单个batch的 model_outputs 和 labels 之间的损失值
        :param model_outputs: 由model_computers.py中 class model_computer_base.model_compute()方法得到的单个batch模型预测值
                              同时也是networks.py文件中class Network.forward()方法的返回值
        :param labels: data_dataset.py中 class dataset.__getitem__()方法中的返回值 labels 真值标签
        :param record: 根据当前batch在整个epoch中的索引和params_dict['TRAIN']['CTRL']['LOSS']['PRINT_GAP'][0]是否整除
                       确定是否保存_loss_init()方法中进行初始化的相关损失值
        :return: model_outputs和labels之间的总损失值，用于反向传播计算梯度、更新权重
        """

        # 自定义各种损失计算, 并将计算结果添加到__loss_init(self)方法中损失列表中去,示例如下:
        # batch_box_loss = self.func0(model_outputs, labels)
        # batch_cls_loss = self.func1(model_outputs, labels)
        # total_loss = weight0 * batch_box_loss + weight1 * batch_cls_loss(weight0和weight1为损失权重,可由params_dict传入)
        # self.box_losses.append(batch_box_loss)
        # self.cls_losses.append(batch_cls_loss)
        # self.total_losses.append(total_loss)

        cls_preds = model_outputs['cls_preds']
        box_preds = model_outputs['box_preds']
        gt_boxes, frame_id = labels
        box_cls_labels, box_reg_targets, reg_weights = self.target_assigner.assign_targets(self.anchors, gt_boxes)

        if not isinstance(cls_preds, list):
            cls_preds = [cls_preds]

        if not isinstance(box_preds, list):
            box_preds = [box_preds]

        batch_size = int(cls_preds[0].shape[0])

        cared = box_cls_labels >= 0
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0

        negative_cls_weights = negatives * self.neg_cls_weight
        positive_cls_weights = positives * self.pos_cls_weight
        cls_weights = (negative_cls_weights + positive_cls_weights).float()
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        one_hot_targets = torch.zeros(*list(cls_targets.shape), self.num_class + 1,
                                      dtype=cls_preds[0].dtype, device=cls_targets.device)
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]

        start_idx = c_idx = 0
        cls_losses = torch.zeros(1, device="cuda:0")
        cls_losses_pos = torch.zeros(1, device='cuda:0')
        cls_losses_list = [0] * len(cls_preds)

        # one_hot_targets = one_hot_targets.cpu()
        # cls_weights = cls_weights.cpu()
        # positive_cls_weights = positive_cls_weights.cpu()
        # box_reg_targets = box_reg_targets.cpu()
        # reg_weights = reg_weights.cpu()
        # pos_normalizer = pos_normalizer.cpu()



        for idx, cls_pred in enumerate(cls_preds):
            one_hot_target = one_hot_targets[:, start_idx:start_idx+cls_pred.shape[1], c_idx:c_idx+1]
            c_idx += 1
            cls_weight = cls_weights[:, start_idx:start_idx+cls_pred.shape[1]]

            cls_loss_fuc = SigmoidFocalLoss()
            cls_loss_src = cls_loss_fuc(cls_pred, [one_hot_target, cls_weight])
            cls_loss = cls_loss_src * cls_weight.unsqueeze(-1)
            cls_loss = (cls_loss * self.cls_weight * self.head_weights[idx]).squeeze(-1)
            cls_loss_pos = cls_loss_src * (positive_cls_weights[:, start_idx:start_idx+cls_pred.shape[1]].float() / torch.clamp(pos_normalizer, min=1.0)).unsqueeze(-1)
            cls_loss_pos = (cls_loss_pos * self.cls_weight * self.head_weights[idx]).squeeze(-1)
            cls_loss = cls_loss.sum() / batch_size
            cls_loss_pos = cls_loss_pos.sum() / batch_size

            cls_losses += cls_loss
            cls_losses_pos += cls_loss_pos
            cls_losses_list[idx] += cls_loss
            start_idx += cls_pred.shape[1]

        assert start_idx == one_hot_targets.shape[1]

        start_idx = 0
        box_losses = torch.zeros(1, device="cuda:0")
        box_losses_list = [0] * len(box_preds)
        for idx, box_pred in enumerate(box_preds):
            box_reg_target = box_reg_targets[:, start_idx:start_idx + box_pred.shape[1]]
            reg_weight = reg_weights[:, start_idx:start_idx + box_pred.shape[1]]

            reg_loss_func = WeightedL1Loss()
            box_loss = reg_loss_func(box_pred, [box_reg_target, reg_weight])

            box_loss = box_loss * self.loc_weight * self.head_weights[idx]
            box_losses += box_loss
            box_losses_list[idx] += box_loss

            start_idx += box_pred.shape[1]

        assert start_idx == box_reg_targets.shape[1]

        total_losses = cls_losses + box_losses

        self.type_cls_losses.append([type_cls_loss.item() for type_cls_loss in cls_losses_list])
        self.type_reg_losses.append([type_reg_loss.item() for type_reg_loss in box_losses_list])
        self.class_losses.append(cls_losses.item())
        self.class_losses_pos.append(cls_losses_pos.item())
        self.box_losses.append(box_losses.item())
        self.total_losses.append(total_losses.item())
        self.latest_metrics = {
            'total_loss': float(total_losses.item()),
            'cls_loss': float(cls_losses.item()),
            'reg_loss': float(box_losses.item()),
            'cls_pos_loss': float(cls_losses_pos.item()),
            'type_cls_loss': [float(type_cls_loss.item()) for type_cls_loss in cls_losses_list],
            'type_reg_loss': [float(type_reg_loss.item()) for type_reg_loss in box_losses_list],
        }

        if record:
            self.loss_record.append([np.mean(self.type_cls_losses, axis=0).tolist(),
                                     np.mean(self.type_reg_losses, axis=0).tolist(),
                                     np.mean(self.class_losses),
                                     np.mean(self.box_losses),
                                     np.mean(self.total_losses),
                                     np.mean(self.class_losses_pos)])
            self.latest_metrics = {
                'total_loss': float(np.mean(self.total_losses)),
                'cls_loss': float(np.mean(self.class_losses)),
                'reg_loss': float(np.mean(self.box_losses)),
                'cls_pos_loss': float(np.mean(self.class_losses_pos)),
                'type_cls_loss': np.mean(self.type_cls_losses, axis=0).tolist(),
                'type_reg_loss': np.mean(self.type_reg_losses, axis=0).tolist(),
            }

            self._loss_initial()

        return total_losses  # 返回的必须是batch数据的总损失值,且tensor.size=1,才能进行反向传播


#     def func0(self, arg0, arg1, ...):
#         "示例：自定义类内方法"
#
#     def func1(self, arg0, arg1, ...):
#         "示例：自定义类内方法"
#
# def function0(arg0, arg1, ...):
#     "自定义类外函数"

class AxisAlignedTargetAssigner(object):
    def __init__(self, model_cfg, class_names, box_coder):
        super().__init__()
        anchor_generator_cfg = model_cfg['ANCHORS']

        self.box_coder = box_coder
        self.class_names = np.array(class_names)
        self.anchor_class_names = [config['class_name'] for config in anchor_generator_cfg]
        self.match_height = False
        self.pos_fraction = None
        self.sample_size = 512
        self.norm_by_num_examples = False
        self.matched_thresholds = {}
        self.unmatched_thresholds = {}
        for config in anchor_generator_cfg:
            self.matched_thresholds[config['class_name']] = config['matched_threshold']
            self.unmatched_thresholds[config['class_name']] = config['unmatched_threshold']

        self.use_multihead = True

    def assign_targets(self, all_anchors, gt_boxes_with_classes):
        """
        Args:
            all_anchors: [(N, 7), ...]list[num_classes=5]:([z=1, y=128, x=128, num_size=1, num_rot=2, 8])
            gt_boxes: (B=3, M, C=8)
        Returns:

        """
        bbox_targets = []
        cls_labels = []
        reg_weights = []

        batch_size = gt_boxes_with_classes.shape[0]
        gt_classes = gt_boxes_with_classes[:, :, -1]
        gt_boxes = gt_boxes_with_classes[:, :, :-1]
        for k in range(batch_size):
            cur_gt = gt_boxes[k]
            cnt = cur_gt.__len__() - 1
            while cnt > 0 and cur_gt[cnt].sum() == 0:  # 舍弃0的box,多帧点云的gt_box拼接时,以gt_box的最大值为该维度的尺寸
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1]
            cur_gt_classes = gt_classes[k][:cnt + 1].int()  # ∈[1,10]

            target_list = []
            # all_anchors：list[5]，对应不同类: ([z=1, y=128, x=128, num_size=1, num_rot=2, 8])
            for anchor_idx, (anchor_class_name, anchors) in enumerate(zip(self.anchor_class_names, all_anchors)):
                mask = (cur_gt_classes - 1) == anchor_idx

                if self.use_multihead:  # True
                    # contiguous()首先拷贝张量的地址，然后将地址按照形状改变后的张量的语义进行排列(为view做准备) ([2*128*128,10])
                    anchors = anchors.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchors.shape[-1])
                    # if self.seperate_multihead:
                    #     selected_classes = cur_gt_classes[mask].clone()
                    #     if len(selected_classes) > 0:
                    #         new_cls_id = self.gt_remapping[anchor_class_name]
                    #         selected_classes[:] = new_cls_id
                    # else:
                    #     selected_classes = cur_gt_classes[mask]
                    selected_classes = cur_gt_classes[mask]
                else:
                    feature_map_size = anchors.shape[:3]
                    anchors = anchors.view(-1, anchors.shape[-1])
                    selected_classes = cur_gt_classes[mask]

                single_target = self.assign_targets_single(
                    anchors,  # ([2*128*128,10])
                    cur_gt[mask],
                    gt_classes=selected_classes,  # car:[1,1,...1]
                    matched_threshold=self.matched_thresholds[anchor_class_name],
                    unmatched_threshold=self.unmatched_thresholds[anchor_class_name]
                )
                target_list.append(single_target)

            if self.use_multihead:
                target_dict = {
                    'box_cls_labels': [t['box_cls_labels'].view(-1) for t in target_list],
                    'box_reg_targets': [t['box_reg_targets'].view(-1, self.box_coder.code_size) for t in target_list],
                    'reg_weights': [t['reg_weights'].view(-1) for t in target_list]
                }

                target_dict['box_reg_targets'] = torch.cat(target_dict['box_reg_targets'], dim=0)  # (32768*C, 10)
                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=0).view(-1)  # (32768*C,)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=0).view(-1)  # (32768*C, 10)
            else:
                target_dict = {
                    'box_cls_labels': [t['box_cls_labels'].view(*feature_map_size, -1) for t in target_list],
                    'box_reg_targets': [t['box_reg_targets'].view(*feature_map_size, -1, self.box_coder.code_size)
                                        for t in target_list],
                    'reg_weights': [t['reg_weights'].view(*feature_map_size, -1) for t in target_list]
                }
                target_dict['box_reg_targets'] = torch.cat(
                    target_dict['box_reg_targets'], dim=-2
                ).view(-1, self.box_coder.code_size)

                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=-1).view(-1)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=-1).view(-1)

            bbox_targets.append(target_dict['box_reg_targets'])
            cls_labels.append(target_dict['box_cls_labels'])
            reg_weights.append(target_dict['reg_weights'])

        bbox_targets = torch.stack(bbox_targets, dim=0)  # (b, 32768*C=5, 8)
        cls_labels = torch.stack(cls_labels, dim=0)  # (b, 32768*C=5)
        reg_weights = torch.stack(reg_weights, dim=0)  # (b, 32768*C=5)

        return cls_labels, bbox_targets, reg_weights
    # anchors:([2*128*128,10]), gt_boxes:([M,8]),一帧点云下为gt_class类别的所有box参数

    def assign_targets_single(self, anchors, gt_boxes, gt_classes, matched_threshold=0.6, unmatched_threshold=0.45):

        num_anchors = anchors.shape[0]  # 2*128*128
        num_gt = gt_boxes.shape[0]

        labels = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1
        gt_ids = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1

        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            # anchor_by_gt_overlap:([2*128*128, M]),只是粗略计算
            anchor_by_gt_overlap = utils.boxes3d_nearest_bev_iou(anchors[:, 0:7], gt_boxes[:, 0:7])

            anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(dim=1)  # ([2*128*128])
            anchor_to_gt_max = anchor_by_gt_overlap[
                torch.arange(num_anchors, device=anchors.device), anchor_to_gt_argmax
            ]  # ([2*128*128,])

            gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(dim=0)  # Indies
            gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, torch.arange(num_gt, device=anchors.device)]
            empty_gt_mask = gt_to_anchor_max == 0
            gt_to_anchor_max[empty_gt_mask] = -1  # ([M, ])

            # 每个gt找到与其IOU最大(非零)的anchors的索引,保证至少有一个anchors与gt匹配 (M,)
            anchors_with_max_overlap = torch.nonzero(anchor_by_gt_overlap == gt_to_anchor_max)[:, 0]
            gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]  # 匹配最大IOU
            gt_ids[anchors_with_max_overlap] = gt_inds_force.int()

            pos_inds = anchor_to_gt_max >= matched_threshold
            gt_inds_over_thresh = anchor_to_gt_argmax[pos_inds]
            labels[pos_inds] = gt_classes[gt_inds_over_thresh]  # anchor与gtIOU>=matched_threshold
            gt_ids[pos_inds] = gt_inds_over_thresh.int()
            bg_inds = (anchor_to_gt_max < unmatched_threshold).nonzero()[:, 0]
        else:
            bg_inds = torch.arange(num_anchors, device=anchors.device)

        fg_inds = (labels > 0).nonzero()[:, 0]

        if self.pos_fraction is not None:  # False
            num_fg = int(self.pos_fraction * self.sample_size)
            if len(fg_inds) > num_fg:
                num_disabled = len(fg_inds) - num_fg
                disable_inds = torch.randperm(len(fg_inds))[:num_disabled]
                labels[disable_inds] = -1
                fg_inds = (labels > 0).nonzero()[:, 0]

            num_bg = self.sample_size - (labels > 0).sum()
            if len(bg_inds) > num_bg:
                enable_inds = bg_inds[torch.randint(0, len(bg_inds), size=(num_bg,))]
                labels[enable_inds] = 0
            # bg_inds = torch.nonzero(labels == 0)[:, 0]
        else:  # True
            if len(gt_boxes) == 0 or anchors.shape[0] == 0:
                labels[:] = 0  # 全部未匹配，表示背景(负样本)
            else:
                labels[bg_inds] = 0  # iou_max<unmatched_threshold,未匹配label设为0，表示背景(负样本)
                labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]

        bbox_targets = anchors.new_zeros((num_anchors, self.box_coder.code_size))  # (63000， 8)
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            fg_gt_boxes = gt_boxes[anchor_to_gt_argmax[fg_inds], :]  # M'*7
            fg_anchors = anchors[fg_inds, :]  # M'*8,M'为最终匹配总数，正样本比例M'/(2*128*128)
            bbox_targets[fg_inds, :] = self.box_coder.encode_torch(fg_gt_boxes, fg_anchors)  # (2*128*128, 8)

        reg_weights = anchors.new_zeros((num_anchors,))

        if self.norm_by_num_examples:
            num_examples = (labels >= 0).sum()
            num_examples = num_examples if num_examples > 1.0 else 1.0
            reg_weights[labels > 0] = 1.0 / num_examples
        else:
            reg_weights[labels > 0] = 1.0

        ret_dict = {
            'box_cls_labels': labels,  # (32768,)
            'box_reg_targets': bbox_targets,  # (32768, 10)，torch.cat([xt, yt, zt, dxt, dyt, dzt, rt_cos, rt_sin, *cts]
            'reg_weights': reg_weights,  # (32768, [labels > 0] = 1.0)，即背景不回归
        }
        return ret_dict


class SigmoidFocalLoss(torch.nn.Module):
    def __init__(self, gamma=4.0, alpha=0.25, reduction='mean'):
        super(SigmoidFocalLoss, self).__init__()
        self._gamma = gamma
        self._alpha = alpha
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target_and_weight: torch.Tensor):
        """
        Args:
            input:
            target_and_weight: 包含了target和obj_weight的元组，原因是因为需要与别的损失函数接口保持统一，不得已而为之
        Returns:

        """
        batch_size = input.shape[0]
        target, obj_weight = target_and_weight
        # 此处交叉熵计算是对tf.nn.sigmoid_cross_entropy_with_logits损失的pytorch版本实现，最主要的是为了防止数据溢出，理论数值上是一致的。
        # https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
        cross_ent_loss = torch.clamp(input, min=0) - input * target.type_as(input) + torch.log1p(torch.exp(-torch.abs(input)))

        pred_probs = torch.sigmoid(input)
        middle_variable = pred_probs * target + (1 - target) * (1 - pred_probs)
        gamma_weight_factor = torch.pow(1.0 - middle_variable, self._gamma)

        alpha_weight_factor = target * self._alpha + (1 - target) * (1 - self._alpha)

        loss = alpha_weight_factor * gamma_weight_factor * cross_ent_loss  # 其实是人为地拆解了focal loss

        # if obj_weight is not None:
        #     loss *= obj_weight.unsqueeze(-1)

        return loss

        # return loss.sum() / batch_size



class WeightedL1Loss(torch.nn.Module):
    def __init__(self, code_weights: list=None):
        # super().__init__()
        super(WeightedL1Loss, self).__init__()
        if code_weights is None:
            self.code_weights = torch.from_numpy(np.ones(8, dtype=np.float32)).cuda()
        else:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    def forward(self, input, target_and_weight):
        """

        Args:
            input: (batch_size, #anchors=128*128*2, #codes(channels)=8) float tensor.
                    Ecoded predicted locations of objects.
            target_and_weight: 包含了target和obj_weight的元组，原因是因为需要与别的损失函数接口保持统一，不得已而为之
                               target: (batch_size, #anchors, #codes) float tensor. Encoded regression targets.
                               weight: (batch_size, #anchors) float tensor if not None.

        Returns:
            loss:(batch_size, #anchors=128*128*2, #codes(channels)=8)
        """
        batch_size = input.shape[0]
        target, weight = target_and_weight
        target = torch.where(torch.isnan(target), input, target)

        diff = input - target
        if self.code_weights is not None:
            diff *= self.code_weights.view(1, 1, -1)

        loss = torch.abs(diff)

        if weight is not None:
            assert weight.shape[0] == loss.shape[0] and weight.shape[1] == loss.shape[1]
            loss *= weight.unsqueeze(-1)

        return loss.sum() / batch_size