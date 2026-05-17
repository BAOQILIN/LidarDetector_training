import torch
import torch.nn as nn


class VFE(nn.Module):
    def __init__(self,
                 voxel_size_x=0.2,
                 voxel_size_y=0.2,
                 voxel__size_z=8.0,
                 range_x_min=-102.4,
                 range_y_min=-51.2,
                 range_z_min=-3.0,
                 use_absolute_xyz=True,
                 with_distance=False):
        super().__init__()
        self.voxel_size_x = voxel_size_x
        self.voxel_size_y = voxel_size_y
        self.voxel_size_z = voxel__size_z
        self.range_x_min = range_x_min
        self.range_y_min = range_y_min
        self.range_z_min = range_z_min
        self.use_absolute_xyz = use_absolute_xyz
        self.with_distance = with_distance
        self.x_offset = self.voxel_size_x / 2 + self.range_x_min
        self.y_offset = self.voxel_size_y / 2 + self.range_y_min
        self.z_offset = self.voxel_size_z / 2 + self.range_z_min

    def forward(self, voxel_features, point_num_per_voxel, voxel_coords):

        point_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / point_num_per_voxel.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - point_mean
        f_center_x = voxel_features[:, :, 0] - (voxel_coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_size_x + self.x_offset)
        f_center_y = voxel_features[:, :, 1] - (voxel_coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_size_y + self.y_offset)
        f_center_z = voxel_features[:, :, 2] - (voxel_coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_size_z + self.z_offset)
        f_center = torch.cat([f_center_x.unsqueeze(2), f_center_y.unsqueeze(2), f_center_z.unsqueeze(2)], dim=2)

        if self.use_absolute_xyz:  # true
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:  # False
            point_dists = torch.norm(voxel_features, p=2, dim=2, keepdim=True)
            features.append(point_dists)

        features = torch.cat(features, dim=-1)

        voxel_cout = features.shape[1]
        mask = torch.unsqueeze(self.get_point_mask(point_num_per_voxel, voxel_cout), dim=-1).type_as(features)
        features *= mask  # [P, 20, 11] 填充点特征全部归零,一个pillar中上限20个点,若真实点数<20,不足者以0填充,是为填充点

        return features

    def get_point_mask(self, point_num_per_voxel, max_num):
        point_num_per_voxel = torch.unsqueeze(point_num_per_voxel, 1).int()
        max_num = torch.arange(max_num, dtype=torch.int, device=point_num_per_voxel.device).view(1, -1)
        point_mask = point_num_per_voxel > max_num
        return point_mask
