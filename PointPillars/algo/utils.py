# 公共类/公共函数定义文件
# 对于其他的7个.py文件中,不允许两两之间互相调用,否则会导致错误
# 如果确实存在彼此公用的功能,务必定义在此文件中
# 其他.py文件可以调用此uitls.py文件

import math
import torch
import numba
import numpy as np
from numba import cuda


def load_ascii_pcd_points(lidar_path):
    data_line_index = None
    with open(lidar_path, 'r', encoding='utf-8', errors='ignore') as f:
        for index, line in enumerate(f):
            if line.strip().lower().startswith('data '):
                data_line_index = index
                break
    if data_line_index is None:
        raise ValueError(f'PCD DATA header not found: {lidar_path}')

    try:
        points = np.loadtxt(lidar_path, skiprows=data_line_index + 1, dtype=np.float32)
    except ValueError as exc:
        raise ValueError(f'Failed to parse ASCII PCD: {lidar_path}') from exc

    if points.size == 0:
        raise ValueError(f'PCD contains no points: {lidar_path}')
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[1] < 4:
        raise ValueError(f'PCD has fewer than 4 columns: {lidar_path}')
    return points[:, :4].astype(np.float32)


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def random_flip_along_x(gt_boxes, points):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 4)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]

    return gt_boxes, points


def random_flip_along_y(gt_boxes, points):  # symmetric about the y-axis
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 4)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        points[:, 0] = -points[:, 0]

    return gt_boxes, points


def global_rotation(gt_boxes, points, rot_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 4),
        rot_range: [min, max]
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    points = rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
    gt_boxes[:, 0:3] = rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
            np.array([noise_rotation])
        )[0][:, 0:2]

    return gt_boxes, points


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa, sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)  # points[:, :, 0:3] * rot_matrix
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def global_scaling(gt_boxes, points, scale_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    return gt_boxes, points


def limit_heading(val, offset=0.5, period=np.pi):
    val, is_numpy = check_numpy_to_torch(val)
    ans = val - torch.floor(val / period + offset) * period
    return ans.numpy() if is_numpy else ans


def points_to_voxel(points,
                    voxel_size=None,
                    point_cloud_range=None,
                    max_num_points=35,
                    reverse_index=True,
                    max_voxels=20000):
    if voxel_size is None:
        voxel_size = [0.2, 0.2, 8]
    if point_cloud_range is None:
        point_cloud_range = [-100, -50.4, -5, 100, 50.4, 3]

    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(point_cloud_range, np.ndarray):
        point_cloud_range = np.array(point_cloud_range, dtype=points.dtype)
    voxel_map_shape = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
    voxel_map_shape = np.round(voxel_map_shape).astype(np.int32)
    voxel_map_shape = tuple(voxel_map_shape.tolist())

    if reverse_index:  # True
        voxel_map_shape = voxel_map_shape[::-1]

    num_pts_per_voxel = np.zeros(shape=(max_voxels,), dtype=np.int32)
    coord_to_voxel_idx = -np.ones(shape=voxel_map_shape, dtype=np.int32)
    voxels = np.zeros(shape=(max_voxels, max_num_points, points.shape[-1]), dtype=points.dtype)
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)

    if reverse_index:  # True
        voxel_num = points_to_voxel_reverse_kernel(points, voxel_size, point_cloud_range,
                                                   num_pts_per_voxel, coord_to_voxel_idx,
                                                   voxels, coors, max_num_points, max_voxels)

    else:
        raise NotImplementedError

    coors = coors[:voxel_num]  # (voxel_num, 3)
    voxels = voxels[:voxel_num]  # (voxel_num, 20, 5)
    num_points_per_voxel = num_pts_per_voxel[:voxel_num]  # (voxel_num, )

    return voxels, coors, num_points_per_voxel


@numba.jit(nopython=True)
def points_to_voxel_reverse_kernel(points, voxel_size, coors_range, num_points_per_voxel,
                                   coord_to_voxel_idx, voxels, coors, max_points=35, max_voxels=20000):
    N = points.shape[0]
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    coor = np.zeros(shape=(3,), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c  # Z Y X 体素坐标
        if failed:
            continue
        voxel_idx = coord_to_voxel_idx[coor[0], coor[1], coor[2]]
        if voxel_idx == -1:
            voxel_idx = voxel_num  # 即第几个体素(<max_voxels)
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            coord_to_voxel_idx[coor[0], coor[1], coor[2]] = voxel_idx  # (1,512,512)
            coors[voxel_idx] = coor  # (max_voxels, 3)
        num = num_points_per_voxel[voxel_idx]
        if num < max_points:
            voxels[voxel_idx, num] = points[i]  # (30000,20,5)
            num_points_per_voxel[voxel_idx] += 1
    return voxel_num


class ResidualCoder(object):
    def __init__(self, code_size=7, encode_angle_by_sincos=False, **kwargs):  # 7, True, **kwargs允许将键值对, 作为参数
        super().__init__()
        self.code_size = code_size  # 7
        self.encode_angle_by_sincos = encode_angle_by_sincos  # True
        if self.encode_angle_by_sincos:
            self.code_size += 1  # 8

    def encode_torch(self, boxes, anchors):
        """
        Args:
            boxes: (N=M', 7 + C=7) [x, y, z, dx, dy, dz, heading]
            anchors: (N=M', 7 + C=8) [x, y, z, dx, dy, dz, heading, 0]

        Returns:

        """
        # 见PointPillars论文,编码只用到了x, y, z, dx, dy, dz, heading, (v1, v2目前不用预测速度)
        anchors[:, 3:6] = torch.clamp_min(anchors[:, 3:6], min=1e-5)  # 限定dx,dy,dz最小值
        boxes[:, 3:6] = torch.clamp_min(boxes[:, 3:6], min=1e-5)

        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)  # (M',1)
        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(boxes, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / dza
        dxt = torch.log(dxg / dxa)
        dyt = torch.log(dyg / dya)
        dzt = torch.log(dzg / dza)
        if self.encode_angle_by_sincos:
            rt_cos = torch.cos(rg) - torch.cos(ra)
            rt_sin = torch.sin(rg) - torch.sin(ra)
            rts = [rt_cos, rt_sin]
        else:
            rts = [rg - ra]

        cts = [g - a for g, a in zip(cgs, cas)]  # 速度信息编码
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, *rts, *cts], dim=-1)

    def decode_torch(self, box_encodings, anchors):
        """
        Args:
            box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        if not self.encode_angle_by_sincos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(box_encodings, 1, dim=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(box_encodings, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(dxt) * dxa
        dyg = torch.exp(dyt) * dya
        dzg = torch.exp(dzt) * dza

        if self.encode_angle_by_sincos:
            rg_cos = cost + torch.cos(ra)
            rg_sin = sint + torch.sin(ra)
            rg = torch.atan2(rg_sin, rg_cos)
        else:
            rg = rt + ra

        cgs = [t + a for t, a in zip(cts, cas)]

        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)

def generate_anchors(anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=8):
    anchor_generator = AnchorGenerator(
        anchor_range=point_cloud_range,
        anchor_generator_config=anchor_generator_cfg)

    # [[128, 128], [128, 128]...],即以4*4(0.2)为一个单元,([512,512,1])->([128,128,1])
    feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in anchor_generator_cfg]
    anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)

    if anchor_ndim != 7:
        for idx, anchors in enumerate(anchors_list):
            pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
            new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
            anchors_list[idx] = new_anchors  # ([z=1, y=128, x=128, num_size=1, num_rot=2, 10])

    return anchors_list, num_anchors_per_location_list
    
class AnchorGenerator(object):
    def __init__(self, anchor_range, anchor_generator_config):
        super().__init__()
        self.anchor_generator_cfg = anchor_generator_config
        self.anchor_range = anchor_range  # point_cloud_range
        self.anchor_sizes = [config['anchor_sizes'] for config in anchor_generator_config]  # (10,1,3)
        self.anchor_rotations = [config['anchor_rotations'] for config in anchor_generator_config]  # (10,2)
        self.anchor_heights = [[config['anchor_bottom_heights']] for config in anchor_generator_config]  # (10,1)
        self.align_center = [config.get('align_center', False) for config in anchor_generator_config]  # [Fasle,...]

        assert len(self.anchor_sizes) == len(self.anchor_rotations) == len(self.anchor_heights)
        self.num_of_anchor_sets = len(self.anchor_sizes)  # 10

    def generate_anchors(self, grid_sizes):  # [[128, 128], [128, 128]...]
        assert len(grid_sizes) == self.num_of_anchor_sets
        all_anchors = []
        num_anchors_per_location = []
        for grid_size, anchor_size, anchor_rotation, anchor_height, align_center in zip(
                grid_sizes, self.anchor_sizes, self.anchor_rotations, self.anchor_heights, self.align_center):  # 0-5

            num_anchors_per_location.append(len(anchor_rotation) * len(anchor_size) * len(anchor_height))  # 2*1*1
            if align_center:
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / grid_size[0]  # 102.4 / 128 = 0.8 = 4 * voxel_size
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / grid_size[1]
                x_offset, y_offset = x_stride / 2, y_stride / 2
            else:
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / (grid_size[0] - 1)  # 0.8063=0.2*4
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / (grid_size[1] - 1)
                x_offset, y_offset = 0, 0

            x_shifts = torch.arange(  # x_shifts, y_shifts, z_shifts 分别表示每个anchor的中心坐标
                self.anchor_range[0] + x_offset, self.anchor_range[3] + 1e-5, step=x_stride, dtype=torch.float32,
            ).cuda()  # ([128])
            y_shifts = torch.arange(
                self.anchor_range[1] + y_offset, self.anchor_range[4] + 1e-5, step=y_stride, dtype=torch.float32,
            ).cuda()
            z_shifts = x_shifts.new_tensor(anchor_height)  # has the same torch.dtype and torch.device as this tensor

            num_anchor_size, num_anchor_rotation = anchor_size.__len__(), anchor_rotation.__len__()  # 1,2
            anchor_rotation = x_shifts.new_tensor(anchor_rotation)  # ([2])
            anchor_size = x_shifts.new_tensor(anchor_size)  # ([1, 3])
            x_shifts, y_shifts, z_shifts = torch.meshgrid([
                x_shifts, y_shifts, z_shifts
            ])  # [x_grid, y_grid, z_grid], x_grid: ([128, 128, 1])
            anchors = torch.stack((x_shifts, y_shifts, z_shifts), dim=-1)  # ([128, 128, 1, 3])
            anchors = anchors[:, :, :, None, :].repeat(1, 1, 1, anchor_size.shape[0], 1)  # ([128, 128, 1, 1, 3])
            anchor_size = anchor_size.view(1, 1, 1, -1, 3).repeat([*anchors.shape[0:3], 1, 1])  # ([128, 128, 1, 1, 3])
            anchors = torch.cat((anchors, anchor_size), dim=-1)  # ([128, 128, 1, 1, 6])
            anchors = anchors[:, :, :, :, None, :].repeat(1, 1, 1, 1, num_anchor_rotation, 1)
            anchor_rotation = anchor_rotation.view(1, 1, 1, 1, -1, 1).repeat([*anchors.shape[0:3], num_anchor_size, 1, 1])
            anchors = torch.cat((anchors, anchor_rotation), dim=-1)  # ([x=128, y=128, z=1, num_size=1, num_rot=2, 7])
            anchors = anchors.permute(2, 1, 0, 3, 4, 5).contiguous()  # 7代表x,y,z,cx,cy,cz,heading
            #anchors = anchors.view(-1, anchors.shape[-1])
            anchors[..., 2] += anchors[..., 5] / 2  # shift to box centers,z=z+h/2

            all_anchors.append(anchors)

        return all_anchors, num_anchors_per_location  # [2,..., 2], 对应0与90度

# =========================================
# IOU计算相关
# =========================================
@numba.jit(nopython=True)
def div_up(m, n):
    return m // n + (m % n > 0)


@cuda.jit('(float32[:], float32[:], float32[:])', device=True, inline=True)
def trangle_area(a, b, c):
    return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) *
            (b[0] - c[0])) / 2.0


@cuda.jit('(float32[:], int32)', device=True, inline=True)
def area(int_pts, num_of_inter):
    area_val = 0.0
    for i in range(num_of_inter - 2):
        area_val += abs(
            trangle_area(int_pts[:2], int_pts[2 * i + 2:2 * i + 4],
                         int_pts[2 * i + 4:2 * i + 6]))
    return area_val


@cuda.jit('(float32[:], int32)', device=True, inline=True)
def sort_vertex_in_convex_polygon(int_pts, num_of_inter):
    if num_of_inter > 0:
        center = cuda.local.array((2,), dtype=numba.float32)
        center[:] = 0.0
        for i in range(num_of_inter):
            center[0] += int_pts[2 * i]
            center[1] += int_pts[2 * i + 1]
        center[0] /= num_of_inter
        center[1] /= num_of_inter
        v = cuda.local.array((2,), dtype=numba.float32)
        vs = cuda.local.array((16,), dtype=numba.float32)
        for i in range(num_of_inter):
            v[0] = int_pts[2 * i] - center[0]
            v[1] = int_pts[2 * i + 1] - center[1]
            d = math.sqrt(v[0] * v[0] + v[1] * v[1])
            v[0] = v[0] / d
            v[1] = v[1] / d
            if v[1] < 0:
                v[0] = -2 - v[0]
            vs[i] = v[0]
        j = 0
        temp = 0
        for i in range(1, num_of_inter):
            if vs[i - 1] > vs[i]:
                temp = vs[i]
                tx = int_pts[2 * i]
                ty = int_pts[2 * i + 1]
                j = i
                while j > 0 and vs[j - 1] > temp:
                    vs[j] = vs[j - 1]
                    int_pts[j * 2] = int_pts[j * 2 - 2]
                    int_pts[j * 2 + 1] = int_pts[j * 2 - 1]
                    j -= 1

                vs[j] = temp
                int_pts[j * 2] = tx
                int_pts[j * 2 + 1] = ty


@cuda.jit(
    '(float32[:], float32[:], int32, int32, float32[:])',
    device=True,
    inline=True)
def line_segment_intersection(pts1, pts2, i, j, temp_pts):
    A = cuda.local.array((2,), dtype=numba.float32)
    B = cuda.local.array((2,), dtype=numba.float32)
    C = cuda.local.array((2,), dtype=numba.float32)
    D = cuda.local.array((2,), dtype=numba.float32)

    A[0] = pts1[2 * i]
    A[1] = pts1[2 * i + 1]

    B[0] = pts1[2 * ((i + 1) % 4)]
    B[1] = pts1[2 * ((i + 1) % 4) + 1]

    C[0] = pts2[2 * j]
    C[1] = pts2[2 * j + 1]

    D[0] = pts2[2 * ((j + 1) % 4)]
    D[1] = pts2[2 * ((j + 1) % 4) + 1]
    BA0 = B[0] - A[0]
    BA1 = B[1] - A[1]
    DA0 = D[0] - A[0]
    CA0 = C[0] - A[0]
    DA1 = D[1] - A[1]
    CA1 = C[1] - A[1]
    acd = DA1 * CA0 > CA1 * DA0
    bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
    if acd != bcd:
        abc = CA1 * BA0 > BA1 * CA0
        abd = DA1 * BA0 > BA1 * DA0
        if abc != abd:
            DC0 = D[0] - C[0]
            DC1 = D[1] - C[1]
            ABBA = A[0] * B[1] - B[0] * A[1]
            CDDC = C[0] * D[1] - D[0] * C[1]
            DH = BA1 * DC0 - BA0 * DC1
            Dx = ABBA * DC0 - BA0 * CDDC
            Dy = ABBA * DC1 - BA1 * CDDC
            temp_pts[0] = Dx / DH
            temp_pts[1] = Dy / DH
            return True
    return False


@cuda.jit(
    '(float32[:], float32[:], int32, int32, float32[:])',
    device=True,
    inline=True)
def line_segment_intersection_v1(pts1, pts2, i, j, temp_pts):
    a = cuda.local.array((2,), dtype=numba.float32)
    b = cuda.local.array((2,), dtype=numba.float32)
    c = cuda.local.array((2,), dtype=numba.float32)
    d = cuda.local.array((2,), dtype=numba.float32)

    a[0] = pts1[2 * i]
    a[1] = pts1[2 * i + 1]

    b[0] = pts1[2 * ((i + 1) % 4)]
    b[1] = pts1[2 * ((i + 1) % 4) + 1]

    c[0] = pts2[2 * j]
    c[1] = pts2[2 * j + 1]

    d[0] = pts2[2 * ((j + 1) % 4)]
    d[1] = pts2[2 * ((j + 1) % 4) + 1]

    area_abc = trangle_area(a, b, c)
    area_abd = trangle_area(a, b, d)

    if area_abc * area_abd >= 0:
        return False

    area_cda = trangle_area(c, d, a)
    area_cdb = area_cda + area_abc - area_abd

    if area_cda * area_cdb >= 0:
        return False
    t = area_cda / (area_abd - area_abc)

    dx = t * (b[0] - a[0])
    dy = t * (b[1] - a[1])
    temp_pts[0] = a[0] + dx
    temp_pts[1] = a[1] + dy
    return True


@cuda.jit('(float32, float32, float32[:])', device=True, inline=True)
def point_in_quadrilateral(pt_x, pt_y, corners):
    ab0 = corners[2] - corners[0]
    ab1 = corners[3] - corners[1]

    ad0 = corners[6] - corners[0]
    ad1 = corners[7] - corners[1]

    ap0 = pt_x - corners[0]
    ap1 = pt_y - corners[1]

    abab = ab0 * ab0 + ab1 * ab1
    abap = ab0 * ap0 + ab1 * ap1
    adad = ad0 * ad0 + ad1 * ad1
    adap = ad0 * ap0 + ad1 * ap1

    return abab >= abap and abap >= 0 and adad >= adap and adap >= 0


@cuda.jit('(float32[:], float32[:], float32[:])', device=True, inline=True)
def quadrilateral_intersection(pts1, pts2, int_pts):
    num_of_inter = 0
    for i in range(4):
        if point_in_quadrilateral(pts1[2 * i], pts1[2 * i + 1], pts2):
            int_pts[num_of_inter * 2] = pts1[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1]
            num_of_inter += 1
        if point_in_quadrilateral(pts2[2 * i], pts2[2 * i + 1], pts1):
            int_pts[num_of_inter * 2] = pts2[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1]
            num_of_inter += 1
    temp_pts = cuda.local.array((2,), dtype=numba.float32)
    for i in range(4):
        for j in range(4):
            has_pts = line_segment_intersection(pts1, pts2, i, j, temp_pts)
            if has_pts:
                int_pts[num_of_inter * 2] = temp_pts[0]
                int_pts[num_of_inter * 2 + 1] = temp_pts[1]
                num_of_inter += 1

    return num_of_inter


@cuda.jit('(float32[:], float32[:])', device=True, inline=True)
def rbbox_to_corners(corners, rbbox):
    # generate clockwise corners and rotate it clockwise
    angle = rbbox[4]
    a_cos = math.cos(angle)
    a_sin = math.sin(angle)
    center_x = rbbox[0]
    center_y = rbbox[1]
    x_d = rbbox[2]
    y_d = rbbox[3]
    corners_x = cuda.local.array((4,), dtype=numba.float32)
    corners_y = cuda.local.array((4,), dtype=numba.float32)
    corners_x[0] = -x_d / 2
    corners_x[1] = -x_d / 2
    corners_x[2] = x_d / 2
    corners_x[3] = x_d / 2
    corners_y[0] = -y_d / 2
    corners_y[1] = y_d / 2
    corners_y[2] = y_d / 2
    corners_y[3] = -y_d / 2
    for i in range(4):  # ego坐标系下
        corners[2 *
                i] = a_cos * corners_x[i] + a_sin * corners_y[i] + center_x
        corners[2 * i
                + 1] = -a_sin * corners_x[i] + a_cos * corners_y[i] + center_y


@cuda.jit('(float32[:], float32[:])', device=True, inline=True)
def inter(rbbox1, rbbox2):
    corners1 = cuda.local.array((8,), dtype=numba.float32)
    corners2 = cuda.local.array((8,), dtype=numba.float32)
    intersection_corners = cuda.local.array((16,), dtype=numba.float32)

    rbbox_to_corners(corners1, rbbox1)  # ego
    rbbox_to_corners(corners2, rbbox2)

    num_intersection = quadrilateral_intersection(corners1, corners2,
                                                  intersection_corners)
    sort_vertex_in_convex_polygon(intersection_corners, num_intersection)
    # print(intersection_corners.reshape([-1, 2])[:num_intersection])

    return area(intersection_corners, num_intersection)


@cuda.jit('(float32[:], float32[:], int32)', device=True, inline=True)
def devRotateIoUEval(rbox1, rbox2, criterion=-1):
    area1 = rbox1[2] * rbox1[3]  #
    area2 = rbox2[2] * rbox2[3]
    area_inter = inter(rbox1, rbox2)
    if criterion == -1:
        return area_inter / (area1 + area2 - area_inter)
    elif criterion == 0:
        return area_inter / area1
    elif criterion == 1:
        return area_inter / area2
    else:
        return area_inter


@cuda.jit('(int64, int64, float32[:], float32[:], float32[:], int32)', fastmath=False)
def rotate_iou_kernel_eval(N, K, dev_boxes, dev_query_boxes, dev_iou, criterion=-1):
    threadsPerBlock = 8 * 8
    row_start = cuda.blockIdx.x
    col_start = cuda.blockIdx.y
    tx = cuda.threadIdx.x
    row_size = min(N - row_start * threadsPerBlock, threadsPerBlock)
    col_size = min(K - col_start * threadsPerBlock, threadsPerBlock)
    block_boxes = cuda.shared.array(shape=(64 * 5,), dtype=numba.float32)
    block_qboxes = cuda.shared.array(shape=(64 * 5,), dtype=numba.float32)

    dev_query_box_idx = threadsPerBlock * col_start + tx
    dev_box_idx = threadsPerBlock * row_start + tx
    if (tx < col_size):
        block_qboxes[tx * 5 + 0] = dev_query_boxes[dev_query_box_idx * 5 + 0]
        block_qboxes[tx * 5 + 1] = dev_query_boxes[dev_query_box_idx * 5 + 1]
        block_qboxes[tx * 5 + 2] = dev_query_boxes[dev_query_box_idx * 5 + 2]
        block_qboxes[tx * 5 + 3] = dev_query_boxes[dev_query_box_idx * 5 + 3]
        block_qboxes[tx * 5 + 4] = dev_query_boxes[dev_query_box_idx * 5 + 4]
    if (tx < row_size):
        block_boxes[tx * 5 + 0] = dev_boxes[dev_box_idx * 5 + 0]
        block_boxes[tx * 5 + 1] = dev_boxes[dev_box_idx * 5 + 1]
        block_boxes[tx * 5 + 2] = dev_boxes[dev_box_idx * 5 + 2]
        block_boxes[tx * 5 + 3] = dev_boxes[dev_box_idx * 5 + 3]
        block_boxes[tx * 5 + 4] = dev_boxes[dev_box_idx * 5 + 4]
    cuda.syncthreads()
    if tx < row_size:
        for i in range(col_size):
            offset = row_start * threadsPerBlock * K + col_start * threadsPerBlock + tx * K + i
            dev_iou[offset] = devRotateIoUEval(block_qboxes[i * 5:i * 5 + 5],
                                               block_boxes[tx * 5:tx * 5 + 5], criterion)


def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    """rotated box iou running in gpu. 500x faster than cpu version
    (take 5ms in one example with numba.cuda code).
    convert from [this project](
        https://github.com/hongzhenwang/RRPN-revise/tree/master/pcdet/rotation).

    Args:
        boxes (float tensor: [N, 5]): rbboxes. format: centers, dims,
            angles(clockwise when positive)
        query_boxes (float tensor: [K, 5]): [description]
        device_id (int, optional): Defaults to 0. [description]

    Returns:
        [type]: [description]
    """
    box_dtype = boxes.dtype
    boxes = boxes.astype(np.float32)
    query_boxes = query_boxes.astype(np.float32)
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    iou = np.zeros((N, K), dtype=np.float32)
    if N == 0 or K == 0:
        return iou
    threadsPerBlock = 8 * 8
    cuda.select_device(device_id)
    blockspergrid = (div_up(N, threadsPerBlock), div_up(K, threadsPerBlock))

    stream = cuda.stream()
    with stream.auto_synchronize():
        boxes_dev = cuda.to_device(boxes.reshape([-1]), stream)
        query_boxes_dev = cuda.to_device(query_boxes.reshape([-1]), stream)
        iou_dev = cuda.to_device(iou.reshape([-1]), stream)
        rotate_iou_kernel_eval[blockspergrid, threadsPerBlock, stream](
            N, K, boxes_dev, query_boxes_dev, iou_dev, criterion)
        iou_dev.copy_to_host(iou.reshape([-1]), stream=stream)
    return iou.astype(boxes.dtype)


def boxes_iou_bev_jit(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7
    boxes1 = torch.cat((boxes_a[:, 0:2], boxes_a[:, 3:5], boxes_a[:, 6:7]), 1)
    boxes2 = torch.cat((boxes_b[:, 0:2], boxes_b[:, 3:5], boxes_b[:, 6:7]), 1)
    iou_bev = rotate_iou_gpu_eval(boxes1.cpu().numpy(), boxes2.cpu().numpy())

    return (torch.from_numpy(iou_bev)).cuda()


def nms_cpu(boxes, scores, threshold=0.5, **kwargs):
    """
    NMS算法
    :param boxes: [N,7]
    :param scores: [N,]
    :param threshold:
    :param kwargs:
    :return:
    """
    assert boxes.shape[1] == 7  # (N,7)
    order = scores.sort(0, descending=True)[1]  # 降序排列,返回序列号 (N,) return indexes of the scores

    keep = []
    while order.numel() > 0:  # torch.numel()返回张量元素个数
        if order.numel() == 1:  # 保留框只剩一个
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()  # 保留scores最大的那个框box[i]
            keep.append(i)

        # 计算box[i]与其余各框的IOU
        boxes_rem = boxes[order[1:]]  # [N-1, 7]
        boxes_max = boxes[i:i + 1, :]  # [1, 7]

        iou = boxes_iou_bev_jit(boxes_max, boxes_rem).squeeze()  # [1, N-1]->[N-1,]
        idx = torch.nonzero(iou <= threshold).squeeze()  # 注意此idx是在boxes_rem中的行索引，也可以认为是order[1:]中的索引
        if idx.numel() == 0:
            break
        order = order[idx + 1]  # 修补索引之间的差值
    return torch.LongTensor(keep).cuda(), None  # Pytorch的索引值为LongTensor


def multi_classes_nms(cls_scores, box_preds, nms_config, score_thresh=None):
    """
    Args:
        cls_scores: (128*128*2*C, C) C=1
        box_preds: (128*128*2*C, 7) C=1
        nms_config:
        score_thresh: 0.2

    Returns:
    """
    pred_scores, pred_labels, pred_boxes = [], [], []
    for k in range(cls_scores.shape[1]):  # for every class in a single head, generally = 1
        if score_thresh is not None:
            scores_mask = (cls_scores[:, k] >= score_thresh)
            box_scores = cls_scores[scores_mask, k]
            cur_box_preds = box_preds[scores_mask]
        else:
            box_scores = cls_scores[:, k]
            cur_box_preds = box_preds

        selected = []
        if box_scores.shape[0] > 0:
            box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.get('nms_pre_maxsize', 1000),
                                                                   box_scores.shape[0]))
            boxes_for_nms = cur_box_preds[indices]

            keep_idx, selected_scores = nms_cpu(
                boxes_for_nms[:, 0:7], box_scores_nms, nms_config.get('nms_thresh', 0.2), **nms_config
            )  # indexes of rest objects of this class
            selected = indices[keep_idx[:nms_config.get('nms_post_maxsize', 83)]]

        pred_scores.append(box_scores[selected])
        pred_labels.append(box_scores.new_ones(len(selected)).long() * k)
        pred_boxes.append(cur_box_preds[selected])

    pred_scores = torch.cat(pred_scores, dim=0)  # [[n,], ...]
    pred_labels = torch.cat(pred_labels, dim=0)  # [[k]*n, ...]
    pred_boxes = torch.cat(pred_boxes, dim=0)  # [[n, 8], ...]  n: number of rest objects in this class

    return pred_scores, pred_labels, pred_boxes


def extract_value(config_dict):
    for k, v in config_dict.items():
        if type(v) is dict:
            extract_value(v)
        else:
            config_dict[k] = v[0]


class BboxGenerator(object):
    def __init__(self, params_dict):
        self.params_dict = params_dict

        self.class_names = ['Pedestrian', 'Mbike', 'Car', 'Bus', 'Tricycle']

        dims = ['X', 'Y', 'Z']
        roi = self.params_dict['TRAIN']['CTRL']['DATA']['ROI']
        point_cloud_range = []
        for dim in dims:
            point_cloud_range.append(roi[f'{dim}_MIN'][0])
        for dim in dims:
            point_cloud_range.append(roi[f'{dim}_MAX'][0])
        self.point_cloud_range = np.array(point_cloud_range, dtype=np.float32)

        voxel_dict = self.params_dict['TRAIN']['CTRL']['DATA']['VOXEL_SIZE']
        voxel_size = []
        for dim in dims:
            voxel_size.append(voxel_dict[f'{dim}'][0])
        self.voxel_size = np.array(voxel_size, dtype=np.float32)

        grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / self.voxel_size
        self.grid_size = np.round(grid_size).astype(np.int64)

        self.code_size = 7

        self.box_coder = ResidualCoder(code_size=7, encode_angle_by_sincos=True)

        anchor_generator_cfg_old = self.params_dict['ANCHORS'].copy()
        anchor_generator_cfg_old = [anchor_generator_cfg_old['ANCHOR_GENERATOR_CONFIG_' + cls_name.upper()] for cls_name in self.class_names]
        for config in anchor_generator_cfg_old:
            extract_value(config)
            anchor_sizes = config['anchor_sizes']
            config['anchor_sizes'] = [[]]
            for dim in dims:
                config['anchor_sizes'][0].append(anchor_sizes[dim])
            config['anchor_rotations'] = [0, 1.57]

        anchor_generator_cfg = []
        for each_cls in self.class_names:
            for each_config in anchor_generator_cfg_old:
                if each_config['class_name'] == each_cls:
                    anchor_generator_cfg.append(each_config)

        self.params_dict['ANCHORS'] = anchor_generator_cfg

        self.anchors, num_anchors_per_location = generate_anchors(anchor_generator_cfg,
                                                                  self.grid_size,
                                                                  self.point_cloud_range,
                                                                  self.code_size)

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds):
        """
        Args:
            batch_size:
            cls_preds: list(len=num_head) of torch.tensor, shape=(b, 128*128*2*C', C') C'=1
            box_preds: list(len=num_head) of torch.tensor, shape=(b, 128*128*2*C', 8) C'=1

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):  # True
            anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                    for anchor in self.anchors], dim=0)  # (128*128*2*num_cls, 8)

        else:  # False
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]  # 128*128*2*num_cls
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)  # (3, 128*128*2*num_cls, 8)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)
        # batch_cls_preds = cls_preds list:5 of torch.tensor, shape=(b, 128*128*2*C', C') C'=1
        # batch_box_preds shape=(b, 128*128*2*num_cls, 8)

        pred_dicts = self.post_processing(batch_cls_preds, batch_box_preds)
        return pred_dicts

    def post_processing(self, batch_cls_preds, batch_box_preds):
        """
        Args:
            batch_cls_preds: list:5 of torch.tensor, shape=(b, 128*128*2*C', C') C'=1
            batch_box_preds: (b, 128*128*2*num_cls, 8)
        Returns:
            pred_dicts  list: batch_size  [dict{str: Tensor}, dict{str: Tensor}, ...]
        """
        nms_cfg = self.params_dict['POSTPROCESS']['NMS']
        score_thresh = self.params_dict['POSTPROCESS']['SCORE_THRESH'][0]
        batch_size = batch_box_preds.shape[0]
        pred_dicts = []
        for index in range(batch_size):
            assert batch_box_preds.shape.__len__() == 3
            batch_mask = index

            box_preds = batch_box_preds[batch_mask]  # (128*128*2*num_cls, 8)

            cls_preds = [x[batch_mask] for x in batch_cls_preds]
            cls_preds = [torch.sigmoid(x) for x in cls_preds]  # list:5 shape=(128*128*2*C', C') C'=1

            multihead_label_mapping = [torch.tensor([i + 1], device=cls_preds[0].device) for i in
                                       range(len(batch_cls_preds))]

            cur_start_idx = 0
            pred_scores, pred_labels, pred_boxes = [], [], []

            for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]

                cur_pred_scores, cur_pred_labels, cur_pred_boxes = multi_classes_nms(
                    cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                    nms_config=nms_cfg,
                    score_thresh=score_thresh
                )
                cur_pred_labels = cur_label_mapping[cur_pred_labels]
                pred_scores.append(cur_pred_scores)
                pred_labels.append(cur_pred_labels)
                pred_boxes.append(cur_pred_boxes)
                cur_start_idx += cur_cls_preds.shape[0]

            final_scores = torch.cat(pred_scores, dim=0)
            final_labels = torch.cat(pred_labels, dim=0)
            final_boxes = torch.cat(pred_boxes, dim=0)

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)

        return pred_dicts

    def boxcoder(self):
        return self.box_coder

    def roi(self):
        return self.point_cloud_range

    def gridsize(self):
        return self.grid_size

    def anchor(self):
        return self.anchors


# =========================================
# aligned_bev_IOU计算相关
# =========================================
def boxes3d_nearest_bev_iou(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:

    """
    boxes_bev_a = boxes3d_lidar_to_aligned_bev_boxes(boxes_a)
    boxes_bev_b = boxes3d_lidar_to_aligned_bev_boxes(boxes_b)

    return boxes_iou_normal(boxes_bev_a, boxes_bev_b)


def boxes3d_lidar_to_aligned_bev_boxes(boxes3d):
    """
    Args:
        boxes3d: (N, 7 + C) [x, y, z, dx, dy, dz, heading] in lidar coordinate

    Returns:
        aligned_bev_boxes: (N, 4) [x_min, y_min, x_max, y_max] in the above lidar coordinate
        指的是 与boxes最接近的 与坐标轴平齐对正的矩形框，以45°为划分界限
    """
    rot_angle = limit_period(boxes3d[:, 6], offset=0.5, period=np.pi).abs()  # ([N,])
    choose_dims = torch.where(rot_angle[:, None] < np.pi / 4, boxes3d[:, [3, 4]], boxes3d[:, [4, 3]])  # ([N,2])
    aligned_bev_boxes = torch.cat((boxes3d[:, 0:2] - choose_dims / 2, boxes3d[:, 0:2] + choose_dims / 2), dim=1)
    return aligned_bev_boxes


def limit_period(val, offset=0.5, period=2 * np.pi):
    if isinstance(val, np.ndarray):
        val, is_numpy = torch.from_numpy(val), True
    else:
        is_numpy = False
    ret = val - torch.floor(val / period + offset) * period
    return ret.numpy() if is_numpy else ret


def boxes_iou_normal(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 4) [x1, y1, x2, y2]
        boxes_b: (M, 4) [x1, y1, x2, y2]

    Returns:

    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 4
    x_min = torch.max(boxes_a[:, 0, None], boxes_b[None, :, 0])  # 0, None额外添加维度
    x_max = torch.min(boxes_a[:, 2, None], boxes_b[None, :, 2])
    y_min = torch.max(boxes_a[:, 1, None], boxes_b[None, :, 1])  # ([N,M])
    y_max = torch.min(boxes_a[:, 3, None], boxes_b[None, :, 3])
    x_len = torch.clamp_min(x_max - x_min, min=0)
    y_len = torch.clamp_min(y_max - y_min, min=0)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    a_intersect_b = x_len * y_len
    iou = a_intersect_b / torch.clamp_min(area_a[:, None] + area_b[None, :] - a_intersect_b, min=1e-6)
    return iou