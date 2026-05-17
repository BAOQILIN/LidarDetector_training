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
        :param pillar_features: torch.size=[P, 20, 64]
        :param pillar_coords:
        :return:
        """
        pillar_features = torch.max(pillar_features, 1, keepdim=True)[0].squeeze()
        batch_spatial_features = []
        batch_size = pillar_coords[:, 0].max().int().item() + 1
        for batch_id in range(batch_size):
            spatial_feature = torch.zeros(self.num_bev_filters,
                                          self.num_gridx * self.num_gridy * self.num_gridz,
                                          dtype=pillar_features.dtype,
                                          device=pillar_features.device)
            batch_mask = pillar_coords[:, 0] == batch_id
            this_coords = pillar_coords[batch_mask, :]
            pillar_indices = this_coords[:, 3] + this_coords[:, 2] * self.num_gridx + this_coords[:, 1]
            pillar_indices = pillar_indices.long()

            this_pillar_features = pillar_features[batch_mask, :].t()
            spatial_feature[:, pillar_indices] = this_pillar_features
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.cat(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_filters * self.num_gridz, self.num_gridx, self.num_gridy)

        return batch_spatial_features

