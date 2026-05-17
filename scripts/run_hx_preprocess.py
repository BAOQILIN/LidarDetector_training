import argparse
import os
import sys

import yaml


def load_config(config_path):
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if 'PREPROCESS' not in cfg:
        raise ValueError(f'Missing PREPROCESS section in config: {config_path}')
    return cfg


def main():
    parser = argparse.ArgumentParser(description='Run HX dataset preprocessing.')
    parser.add_argument('--data-root', required=True, help='HX dataset root, e.g. /home/bql/ARS/ARS_Data/ars_hx_train_data')
    parser.add_argument('--config', default='PointPillars/algo/algo_config_hx.yaml', help='Path to HX config file')
    parser.add_argument('--mode', choices=['train', 'test', 'predict'], default='train', help='Which pickle set to generate')
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = args.config if os.path.isabs(args.config) else os.path.join(repo_root, args.config)

    sys.path.insert(0, os.path.join(repo_root, 'PointPillars/algo'))
    from data_preprocessor import DataPreprocessor

    cfg = load_config(config_path)
    preprocess_cfg = cfg['PREPROCESS']
    res = {'msg': []}

    train_flag = args.mode == 'train'
    test_flag = args.mode == 'test'
    preprocessor = DataPreprocessor(preprocess_cfg, args.data_root, train_flag=train_flag, test_flag=test_flag, res_dict=res)
    preprocessor.data_preprocess()

    for message in res['msg']:
        print(message)


if __name__ == '__main__':
    main()
