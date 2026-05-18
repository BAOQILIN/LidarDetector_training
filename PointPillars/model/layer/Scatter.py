import torch
import torch.nn as nn


class Scatter(nn.Module):
    def __init__(self, num_filters=64, grid_size=None, **kwargs):
        super().__init__()
        self.num_bev_filters = num_filters
        if grid_size is None:
            grid_size = [1024, 512, 1]
        self.num_gridx, self.num_gridy, self.num_gridz = grid_size
        assert self.num_gridz == 1

    def forward(self, pillar_features, pillar_coords):
        """
        :param pillar_features: torch.size=[P, max_points, C]
        :param pillar_coords:   torch.size=[P, 4]  [batch_idx, z, y, x]
        :return: batch_spatial_features: (B, C*nz, nx, ny)
        """
        pillar_features = torch.max(pillar_features, 1, keepdim=True)[0].squeeze()  # (P, C)
        batch_size = pillar_coords[:, 0].max().int().item() + 1

        batch_idx = pillar_coords[:, 0].long()
        z = pillar_coords[:, 1].long()
        y = pillar_coords[:, 2].long()
        x = pillar_coords[:, 3].long()

        # per-batch 1D spatial index: z * (ny * nx) + y * nx + x
        cells_per_sample = self.num_gridx * self.num_gridy * self.num_gridz
        spatial_idx = z * (self.num_gridx * self.num_gridy) + y * self.num_gridx + x
        global_idx = batch_idx * cells_per_sample + spatial_idx

        total_cells = batch_size * cells_per_sample
        spatial_feature = torch.zeros(
            self.num_bev_filters, total_cells,
            dtype=pillar_features.dtype, device=pillar_features.device)
        spatial_feature[:, global_idx] = pillar_features.t()

        batch_spatial_features = spatial_feature.view(
            batch_size, self.num_bev_filters * self.num_gridz,
            self.num_gridx, self.num_gridy)
        return batch_spatial_features
