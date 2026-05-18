#!/usr/bin/env python3
"""
run_hx_eval.py - HX PointPillars model evaluation and visualization.

Usage:
  # Single-frame visualization with Open3D (GT=green, Pred=red)
  python scripts/run_hx_eval.py \
    --data-root /path/to/hx_data \
    --result-root /path/to/results \
    --epoch 10 \
    --visualize

  # Visualize a specific frame by index
  python scripts/run_hx_eval.py \
    --data-root /path/to/hx_data \
    --result-root /path/to/results \
    --epoch 10 \
    --visualize \
    --index 42

  # Batch evaluation on the test set
  python scripts/run_hx_eval.py \
    --data-root /path/to/hx_data \
    --result-root /path/to/results \
    --epoch 10 \
    --evaluate

  # Both: evaluate then visualize the first frame
  python scripts/run_hx_eval.py \
    --data-root /path/to/hx_data \
    --result-root /path/to/results \
    --epoch 10 \
    --evaluate \
    --visualize
"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import yaml

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'PointPillars', 'algo'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'PointPillars', 'model', 'model'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'PointPillars', 'model', 'layer'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'web_lidardetector'))

from train_flow import LidarDetector
from utils import load_ascii_pcd_points, BboxGenerator, points_to_voxel_gpu


CLASS_NAMES = ['Pedestrian', 'Mbike', 'Car', 'Bus', 'Tricycle']
GT_COLOR = (0.0, 1.0, 0.0)    # green
PRED_COLOR = (1.0, 0.0, 0.0)  # red


def load_config(config_path):
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if 'TRAIN_MODEL' not in cfg:
        raise ValueError(f'Missing TRAIN_MODEL section: {config_path}')
    return cfg


def make_detector(cfg, data_root, result_root, model_epoch_root, dataset_type):
    train_cfg = copy.deepcopy(cfg['TRAIN_MODEL'])
    train_flag = dataset_type == 'train'
    test_flag = dataset_type == 'test'
    return LidarDetector(
        params_dict=train_cfg,
        data_root=data_root,
        result_root=result_root,
        model_epoch_root=model_epoch_root,
        train_flag=train_flag,
        test_flag=test_flag,
    )


def boxes_to_corners(boxes):
    """Convert boxes [x, y, z, dx, dy, dz, heading] to 8 corners (N, 8, 3)."""
    if boxes.shape[0] == 0:
        return np.zeros((0, 8, 3), dtype=np.float32)

    x, y, z = boxes[:, 0], boxes[:, 1], boxes[:, 2]
    dx, dy, dz = boxes[:, 3], boxes[:, 4], boxes[:, 5]
    heading = boxes[:, 6]

    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    half_dx, half_dy = dx / 2.0, dy / 2.0

    corners = np.zeros((boxes.shape[0], 8, 3), dtype=np.float32)
    for i, (sx, sy) in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
        lx = sx * half_dx
        ly = sy * half_dy
        rx = cos_h * lx - sin_h * ly
        ry = sin_h * lx + cos_h * ly
        corners[:, i, 0] = x + rx
        corners[:, i, 1] = y + ry
        corners[:, i, 2] = z - dz / 2.0
        corners[:, i + 4, 0] = x + rx
        corners[:, i + 4, 1] = y + ry
        corners[:, i + 4, 2] = z + dz / 2.0

    return corners


def create_box_lineset(corners, color):
    """Create Open3D LineSet from box corners. Returns None if no boxes."""
    if corners.shape[0] == 0:
        return None

    import open3d as o3d

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    all_points = []
    all_lines = []
    offset = 0
    for i in range(corners.shape[0]):
        for j in range(8):
            all_points.append(corners[i, j])
        for e in edges:
            all_lines.append([offset + e[0], offset + e[1]])
        offset += 8

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(np.array(all_points, dtype=np.float64))
    lineset.lines = o3d.utility.Vector2iVector(np.array(all_lines, dtype=np.int32))
    line_colors = np.tile(np.array(color, dtype=np.float64), (len(all_lines), 1))
    lineset.colors = o3d.utility.Vector3dVector(line_colors)
    return lineset


def create_point_cloud(points):
    """Create Open3D PointCloud with uniform blue coloring."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
    colors = np.tile(np.array([[0.0, 0.2, 0.8]], dtype=np.float64), (points.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def run_single_inference(model, points, voxel_params, bbox_generator):
    """Run inference on a single sample and return pred_dict.

    Voxelization is done on GPU from raw points.
    """
    import torch

    voxels, coors, num_pts = points_to_voxel_gpu(
        points,
        voxel_params['voxel_size'],
        voxel_params['point_cloud_range'],
        voxel_params['max_num_points'],
        voxel_params['max_voxels'],
    )
    coors = torch.cat([
        torch.zeros(coors.shape[0], 1, dtype=torch.int32, device=coors.device),
        coors,
    ], dim=1).float()
    num_pts = num_pts.float()

    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            outputs = model([voxels, num_pts, coors])

    pred_dicts = bbox_generator.generate_predicted_boxes(
        batch_size=1,
        cls_preds=outputs['cls_preds'],
        box_preds=outputs['box_preds'],
    )
    return pred_dicts[0]


def _check_dataset(detector, dataset_key, purpose):
    """Verify the dataset is non-empty, or raise with a helpful message."""
    ds = detector.data_component.data_loader.dataset.get(dataset_key)
    if ds is None or len(ds) == 0:
        dataset_name = {'eva_test': 'testing.pkl', 'eva_predict': 'prediction.pkl'}.get(dataset_key, dataset_key)
        prefix_map = {'eva_test': 'test', 'eva_predict': 'predict'}
        mode = prefix_map.get(dataset_key, 'test')
        raise RuntimeError(
            f'Dataset "{dataset_key}" is empty — {dataset_name} not found or has no samples.\n'
            f'Run preprocessing first:\n'
            f'  python scripts/run_hx_preprocess.py \\\n'
            f'    --data-root <data_root> \\\n'
            f'    --config <config> \\\n'
            f'    --mode {mode}\n'
        )


def visualize_frame(detector, dataset_key, index, score_thresh):
    """Show a single frame in Open3D: point cloud + GT (green) + pred boxes (red)."""
    import open3d as o3d

    _check_dataset(detector, dataset_key, 'visualization')
    ds = detector.data_component.data_loader.dataset[dataset_key]
    if index < 0 or index >= len(ds):
        raise IndexError(f'Index {index} out of range [0, {len(ds) - 1}] for dataset "{dataset_key}"')

    sample = ds.infos[index]
    lidar_path = sample['lidar_path']
    gt_boxes_raw = sample['gt_boxes'].astype(np.float32)
    gt_names = sample['gt_names']

    print(f'Frame #{index}: {lidar_path}')
    print(f'  GT objects: {len(gt_boxes_raw)}')
    for i, (name, box) in enumerate(zip(gt_names, gt_boxes_raw)):
        print(f'    [{i}] {name} @ ({box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f})')

    raw_points = load_ascii_pcd_points(lidar_path)

    data = ds[index]
    model = detector.model_component.model_computer.model

    bbox_generator = BboxGenerator(copy.deepcopy(detector.params_dict))
    voxel_params = {
        'voxel_size': ds.voxel_size,
        'point_cloud_range': ds.point_cloud_range,
        'max_num_points': ds.max_num_points,
        'max_voxels': ds.max_voxels,
    }
    pred_dict = run_single_inference(
        model, data['points'], voxel_params, bbox_generator
    )

    pred_boxes = pred_dict['pred_boxes'].cpu().numpy()
    pred_scores = pred_dict['pred_scores'].cpu().numpy()
    pred_labels = pred_dict['pred_labels'].cpu().numpy()

    if score_thresh > 0:
        keep = pred_scores >= score_thresh
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]

    print(f'  Pred objects (score >= {score_thresh}): {len(pred_boxes)}')
    for i in range(len(pred_boxes)):
        cls_idx = int(pred_labels[i]) - 1
        cls_name = CLASS_NAMES[cls_idx] if 0 <= cls_idx < len(CLASS_NAMES) else f'cls_{pred_labels[i]}'
        box = pred_boxes[i]
        print(f'    [{i}] {cls_name} score={pred_scores[i]:.3f} @ ({box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f})')

    geoms = [create_point_cloud(raw_points)]

    gt_corners = boxes_to_corners(gt_boxes_raw[:, :7])
    gt_ls = create_box_lineset(gt_corners, GT_COLOR)
    if gt_ls is not None:
        geoms.append(gt_ls)

    pred_corners = boxes_to_corners(pred_boxes[:, :7])
    pred_ls = create_box_lineset(pred_corners, PRED_COLOR)
    if pred_ls is not None:
        geoms.append(pred_ls)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=[0, 0, 0])
    geoms.append(axis)

    viewer = o3d.visualization.Visualizer()
    viewer.create_window(window_name=f'HX Eval - Frame #{index}  |  GT: green  Pred: red')
    for g in geoms:
        viewer.add_geometry(g)

    opt = viewer.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])
    opt.point_size = 1.0
    opt.line_width = 2.0

    viewer.run()
    viewer.destroy_window()


def run_evaluation(detector, epoch_list):
    """Run full test-set evaluation and save results."""
    _check_dataset(detector, 'eva_test', 'evaluation')
    detector.test(epoch_list, save=True)


def main():
    parser = argparse.ArgumentParser(description='HX PointPillars evaluation and visualization')
    parser.add_argument('--data-root', required=True, help='HX dataset root')
    parser.add_argument('--result-root', required=True, help='Training result directory')
    parser.add_argument('--config', default='PointPillars/algo/algo_config_hx.yaml', help='Config file path')
    parser.add_argument('--epoch', type=int, required=True, help='Model epoch to evaluate')
    parser.add_argument('--visualize', action='store_true', help='Single-frame visualization with Open3D')
    parser.add_argument('--evaluate', action='store_true', help='Batch evaluation on test set')
    parser.add_argument('--index', type=int, default=0, help='Frame index for visualization (default: 0)')
    parser.add_argument('--score-thresh', type=float, default=0.3, help='Score threshold for pred boxes (default: 0.2)')
    parser.add_argument('--dataset', choices=['test', 'predict'], default='test',
                        help='Dataset for visualization (default: test)')
    args = parser.parse_args()

    if not args.visualize and not args.evaluate:
        parser.error('Must specify at least one of --visualize or --evaluate')

    config_path = args.config if os.path.isabs(args.config) else os.path.join(_REPO_ROOT, args.config)
    cfg = load_config(config_path)
    model_epoch_root = os.path.join(args.result_root, 'model_epoch')

    if args.evaluate:
        print('Loading model for evaluation ...')
        eval_detector = make_detector(cfg, args.data_root, args.result_root, model_epoch_root, 'test')
        eval_detector.model_component.model_computer.load_model_params_torch(args.epoch)
        print(f'Loaded model epoch {args.epoch}')
        print('Starting batch evaluation on test set ...')
        t_start = time.time()
        run_evaluation(eval_detector, [args.epoch])
        print(f'Evaluation done in {time.time() - t_start:.1f}s')
        eva_path = os.path.join(args.result_root, 'loss_eva', 'eva_test.npy')
        if os.path.exists(eva_path):
            print(f'Results saved to {eva_path}')

    if args.visualize:
        print('Loading model for visualization ...')
        viz_detector = make_detector(cfg, args.data_root, args.result_root, model_epoch_root, args.dataset)
        viz_detector.model_component.model_computer.load_model_params_torch(args.epoch)
        print(f'Loaded model epoch {args.epoch}')
        ds_key = {'test': 'eva_test', 'predict': 'eva_predict'}[args.dataset]
        _check_dataset(viz_detector, ds_key, 'visualization')
        print(f'Opening visualization for frame #{args.index} ...')
        visualize_frame(viz_detector, ds_key, args.index, args.score_thresh)


if __name__ == '__main__':
    main()
