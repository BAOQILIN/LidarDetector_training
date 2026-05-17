import argparse
import os
import sys

import yaml


def load_config(config_path):
    with open(config_path, encoding='utf-8') as file:
        cfg = yaml.safe_load(file)
    if 'TRAIN_MODEL' not in cfg:
        raise ValueError(f'Missing TRAIN_MODEL section in config: {config_path}')
    return cfg


def main():
    parser = argparse.ArgumentParser(description='Export HX PointPillars checkpoint to ONNX files.')
    parser.add_argument('--result-root', required=True, help='Training result directory that contains model_epoch')
    parser.add_argument('--config', default='PointPillars/algo/algo_config_hx.yaml', help='Path to HX config file')
    parser.add_argument('--epoch', type=int, default=None, help='Checkpoint epoch to export')
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = args.config if os.path.isabs(args.config) else os.path.join(repo_root, args.config)
    model_epoch_root = os.path.join(args.result_root, 'model_epoch')

    sys.path.insert(0, os.path.join(repo_root, 'PointPillars/algo'))
    sys.path.insert(0, os.path.join(repo_root, 'PointPillars/model/model'))
    sys.path.insert(0, os.path.join(repo_root, 'PointPillars/model/layer'))
    sys.path.insert(0, os.path.join(repo_root, 'web_lidardetector'))
    from ModelUtils.model_components import model_components

    cfg = load_config(config_path)
    train_cfg = cfg['TRAIN_MODEL']
    save_epoch = args.epoch if args.epoch is not None else train_cfg['SAVE']['SAVE_EPOCH'][0]

    checkpoint_path = os.path.join(model_epoch_root, f"{train_cfg['TRAIN']['PATH']['SAVE_MODEL_PREFIX'][0]}_{save_epoch}.torch")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f'Missing checkpoint for epoch {save_epoch}: {checkpoint_path}. '
            f'Please confirm the epoch number and result-root.'
        )

    res = {'msg': []}
    model_component = model_components(train_cfg, args.result_root, model_epoch_root, pretrained_path=None, res_dict=res)
    model_component.model_computer.save_model_params_onnx(save_epoch)

    for message in res['msg']:
        print(message)

    export_dir = os.path.join(model_epoch_root, f"{train_cfg['TRAIN']['PATH']['SAVE_MODEL_PREFIX'][0]}_epoch_{save_epoch}")
    print(f'Exported ONNX files to: {export_dir}')


if __name__ == '__main__':
    main()
