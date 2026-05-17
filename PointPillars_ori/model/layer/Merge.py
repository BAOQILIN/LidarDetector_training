import torch.nn as nn


class Merge(nn.Module):
    def __init__(self):
        super(Merge, self).__init__()
        self.res_dicts = []

    def forward(self, input1, input2, input3, input4, input5):
        self.res_dicts = [input1, input2, input3, input4, input5]
        cls_preds = [res_dict['cls_preds'] for res_dict in self.res_dicts]
        box_preds = [res_dict['box_preds'] for res_dict in self.res_dicts]

        ret = {'cls_preds': cls_preds, 'box_preds': box_preds}

        return ret

