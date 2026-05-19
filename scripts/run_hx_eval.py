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
POINT_COLOR = (0.0, 0.2, 0.8)  # blue
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


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

    all_points = []
    all_lines = []
    offset = 0
    for i in range(corners.shape[0]):
        for j in range(8):
            all_points.append(corners[i, j])
        for e in BOX_EDGES:
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
    colors = np.tile(np.array([POINT_COLOR], dtype=np.float64), (points.shape[0], 1))
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


def class_name_from_label(label):
    cls_idx = int(label) - 1
    return CLASS_NAMES[cls_idx] if 0 <= cls_idx < len(CLASS_NAMES) else f'cls_{label}'


def build_frame_data(detector, dataset_key, index, score_thresh):
    _check_dataset(detector, dataset_key, 'visualization')
    ds = detector.data_component.data_loader.dataset[dataset_key]
    if index < 0 or index >= len(ds):
        raise IndexError(f'Index {index} out of range [0, {len(ds) - 1}] for dataset "{dataset_key}"')

    sample = ds.infos[index]
    lidar_path = sample['lidar_path']
    gt_boxes = sample['gt_boxes'].astype(np.float32)
    gt_names = sample['gt_names']
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
    pred_dict = run_single_inference(model, data['points'], voxel_params, bbox_generator)

    pred_boxes = pred_dict['pred_boxes'].cpu().numpy()
    pred_scores = pred_dict['pred_scores'].cpu().numpy()
    pred_labels = pred_dict['pred_labels'].cpu().numpy()

    if score_thresh > 0:
        keep = pred_scores >= score_thresh
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]

    gt_labels = [
        {
            'text': str(name),
            'position': np.array([box[0], box[1], box[2] + box[5] / 2.0 + 0.3], dtype=np.float32),
            'color': GT_COLOR,
        }
        for name, box in zip(gt_names, gt_boxes)
    ]
    pred_labels_info = [
        {
            'text': f"{class_name_from_label(label)} {score:.2f}",
            'position': np.array([box[0], box[1], box[2] + box[5] / 2.0 + 0.3], dtype=np.float32),
            'color': PRED_COLOR,
        }
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores)
    ]

    return {
        'index': index,
        'lidar_path': lidar_path,
        'raw_points': raw_points,
        'gt_boxes': gt_boxes,
        'gt_names': gt_names,
        'pred_boxes': pred_boxes,
        'pred_scores': pred_scores,
        'pred_labels': pred_labels,
        'gt_label_infos': gt_labels,
        'pred_label_infos': pred_labels_info,
    }


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


def _print_frame_summary(frame_data, score_thresh):
    print(f"Frame #{frame_data['index']}: {frame_data['lidar_path']}")
    print(f"  GT objects: {len(frame_data['gt_boxes'])}")
    for i, (name, box) in enumerate(zip(frame_data['gt_names'], frame_data['gt_boxes'])):
        print(f'    [{i}] {name} @ ({box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f})')

    print(f"  Pred objects (score >= {score_thresh}): {len(frame_data['pred_boxes'])}")
    for i, (box, label, score) in enumerate(zip(frame_data['pred_boxes'], frame_data['pred_labels'], frame_data['pred_scores'])):
        print(f'    [{i}] {class_name_from_label(label)} score={score:.3f} @ ({box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f})')


class EvalFrameViewer:
    def __init__(self, detector, dataset_key, start_index, score_thresh):
        import open3d as o3d

        self.o3d = o3d
        self.detector = detector
        self.dataset_key = dataset_key
        self.score_thresh = score_thresh
        self.dataset = detector.data_component.data_loader.dataset[dataset_key]
        self.index = start_index
        self.app = o3d.visualization.gui.Application.instance
        self.window = None
        self.scene_widget = None
        self.scene = None
        self.materials = {}
        self.axis = None
        self.current_bounds = None
        self.label_handles = []

    def run(self):
        self.app.initialize()
        self.window = self.app.create_window('HX Eval Viewer', 1600, 900)
        self.window.set_on_close(self._on_close)
        self.scene_widget = self.o3d.visualization.gui.SceneWidget()
        self.scene_widget.scene = self.o3d.visualization.rendering.Open3DScene(self.window.renderer)
        self.scene = self.scene_widget.scene
        self.scene.set_background([0.1, 0.1, 0.1, 1.0])
        self.scene_widget.set_on_key(self._on_key)
        self.window.add_child(self.scene_widget)
        self.window.set_focus_widget(self.scene_widget)
        self.window.set_on_layout(self._on_layout)
        self._init_materials()
        self.axis = self.o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=[0, 0, 0])
        self._load_frame(self.index, reset_camera=True)
        self.app.run()

    def _init_materials(self):
        unlit_line = self.o3d.visualization.rendering.MaterialRecord()
        unlit_line.shader = 'unlitLine'
        unlit_line.line_width = 2.0

        unlit_point = self.o3d.visualization.rendering.MaterialRecord()
        unlit_point.shader = 'defaultUnlit'
        unlit_point.point_size = 1.0

        axis_material = self.o3d.visualization.rendering.MaterialRecord()
        axis_material.shader = 'defaultUnlit'

        self.materials = {
            'point_cloud': unlit_point,
            'gt_boxes': unlit_line,
            'pred_boxes': unlit_line,
            'axis': axis_material,
        }

    def _on_layout(self, layout_context):
        self.scene_widget.frame = self.window.content_rect

    def _on_close(self):
        return True

    def _on_key(self, event):
        key = self.o3d.visualization.gui.KeyName
        event_type = self.o3d.visualization.gui.KeyEvent.Type
        result = self.o3d.visualization.gui.SceneWidget.EventCallbackResult
        if event.type != event_type.DOWN:
            return result.IGNORED

        if event.key in (key.A, key.LEFT):
            self._load_frame(self.index - 1)
            self.window.set_focus_widget(self.scene_widget)
            return result.HANDLED
        if event.key in (key.D, key.RIGHT):
            self._load_frame(self.index + 1)
            self.window.set_focus_widget(self.scene_widget)
            return result.HANDLED
        return result.IGNORED

    def _update_window_title(self, frame_data):
        self.window.title = (
            f"HX Eval - Frame #{frame_data['index'] + 1}/{len(self.dataset)}"
            '  |  A/←: prev  D/→: next  |  GT: green  Pred: red'
        )

    def _clear_scene(self):
        self.scene.clear_geometry()
        for label_handle in self.label_handles:
            self.scene_widget.remove_3d_label(label_handle)
        self.label_handles.clear()

    def _compute_bounds(self, frame_data):
        points = frame_data['raw_points'][:, :3]
        if points.size > 0:
            min_bound = points.min(axis=0)
            max_bound = points.max(axis=0)
        else:
            min_bound = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
            max_bound = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        for boxes in (frame_data['gt_boxes'], frame_data['pred_boxes']):
            if boxes.shape[0] == 0:
                continue
            corners = boxes_to_corners(boxes[:, :7]).reshape(-1, 3)
            min_bound = np.minimum(min_bound, corners.min(axis=0))
            max_bound = np.maximum(max_bound, corners.max(axis=0))

        bounds = self.o3d.geometry.AxisAlignedBoundingBox(
            min_bound.astype(np.float64),
            max_bound.astype(np.float64),
        )
        if np.any(bounds.get_extent() <= 1e-6):
            bounds = self.o3d.geometry.AxisAlignedBoundingBox(
                min_bound.astype(np.float64) - 1.0,
                max_bound.astype(np.float64) + 1.0,
            )
        return bounds

    def _reset_camera(self, bounds):
        self.scene_widget.setup_camera(60.0, bounds, bounds.get_center())

    def _load_frame(self, index, reset_camera=False):
        bounded_index = max(0, min(index, len(self.dataset) - 1))
        if bounded_index == self.index and self.current_bounds is not None and not reset_camera:
            return

        frame_data = build_frame_data(self.detector, self.dataset_key, bounded_index, self.score_thresh)
        self.index = bounded_index
        self._clear_scene()
        self._update_window_title(frame_data)
        _print_frame_summary(frame_data, self.score_thresh)

        point_cloud = create_point_cloud(frame_data['raw_points'])
        self.scene.add_geometry('point_cloud', point_cloud, self.materials['point_cloud'])

        gt_corners = boxes_to_corners(frame_data['gt_boxes'][:, :7])
        gt_ls = create_box_lineset(gt_corners, GT_COLOR)
        if gt_ls is not None:
            self.scene.add_geometry('gt_boxes', gt_ls, self.materials['gt_boxes'])

        pred_corners = boxes_to_corners(frame_data['pred_boxes'][:, :7])
        pred_ls = create_box_lineset(pred_corners, PRED_COLOR)
        if pred_ls is not None:
            self.scene.add_geometry('pred_boxes', pred_ls, self.materials['pred_boxes'])

        self.scene.add_geometry('axis', self.axis, self.materials['axis'])

        for label_info in frame_data['gt_label_infos']:
            label = self.scene_widget.add_3d_label(label_info['position'], label_info['text'])
            label.color = self.o3d.visualization.gui.Color(*label_info['color'])
            self.label_handles.append(label)
        for label_info in frame_data['pred_label_infos']:
            label = self.scene_widget.add_3d_label(label_info['position'], label_info['text'])
            label.color = self.o3d.visualization.gui.Color(*label_info['color'])
            self.label_handles.append(label)

        bounds = self._compute_bounds(frame_data)
        if reset_camera or self.current_bounds is None:
            self._reset_camera(bounds)
        self.current_bounds = bounds
        self.window.set_focus_widget(self.scene_widget)
        self.window.post_redraw()


def visualize_frame(detector, dataset_key, index, score_thresh):
    """Show an interactive Open3D viewer with labels and frame navigation."""
    _check_dataset(detector, dataset_key, 'visualization')
    viewer = EvalFrameViewer(detector, dataset_key, index, score_thresh)
    viewer.run()


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
