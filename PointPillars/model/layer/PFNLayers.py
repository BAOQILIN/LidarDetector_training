import torch
import torch.nn as nn
import torch.nn.functional as F


class PFNLayers(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.linear = nn.Conv1d(input_channels, output_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm1d(output_channels, eps=1e-3, momentum=0.01)
        self.limit = 50000

    def forward(self, x):
        inputs = x
        inputs = inputs.transpose(1, 2)
        if inputs.shape[0] > self.limit:
            num_part = inputs.shape[0] // self.limit
            part_linear_out = [self.linear(inputs[i * self.limit:(i + 1) * self.limit]) for i in range(num_part + 1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)

        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        x = self.norm(x).permute(0, 2, 1)
        torch.backends.cudnn.enabled = cudnn_enabled
        x = F.relu(x)

        return x
