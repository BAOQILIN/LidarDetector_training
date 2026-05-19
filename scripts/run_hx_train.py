import argparse
import os
import sys
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import yaml


def load_config(config_path):
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if 'TRAIN_MODEL' not in cfg:
        raise ValueError(f'Missing TRAIN_MODEL section in config: {config_path}')
    if 'PREPROCESS' not in cfg:
        raise ValueError(f'Missing PREPROCESS section in config: {config_path}')
    return cfg


def main():
    parser = argparse.ArgumentParser(description='Run HX PointPillars training.')
    parser.add_argument('--data-root', required=True, help='HX dataset root, e.g. /home/bql/ARS/ARS_Data/ars_hx_train_data')
    parser.add_argument('--result-root', required=True, help='Directory for training results')
    parser.add_argument('--config', default='PointPillars/algo/algo_config_hx.yaml', help='Path to HX config file')
    parser.add_argument('--smoke', action='store_true', help='Run a 1-epoch smoke test with batch size 1 and check_flag enabled')
    parser.add_argument('--epochs', type=int, default=None, help='Override epoch count')
    parser.add_argument('--batch-size', '--batch_size', dest='batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--eval-every-epochs', type=int, default=None, help='Override validation frequency in epochs')
    parser.add_argument('--num-workers', type=int, default=None, help='Override DataLoader worker count for train/eval')
    parser.add_argument('--multi-gpu', action='store_true', help='Enable DataParallel when multiple CUDA devices are available')
    parser.add_argument('--resume-epoch', type=int, default=None, help='Resume from checkpoint index N (0-based, loads *_N.torch) and continue from the next epoch')
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = args.config if os.path.isabs(args.config) else os.path.join(repo_root, args.config)
    model_epoch_root = os.path.join(args.result_root, 'model_epoch')

    sys.path.insert(0, os.path.join(repo_root, 'PointPillars/algo'))
    sys.path.insert(0, os.path.join(repo_root, 'PointPillars/model/model'))
    sys.path.insert(0, os.path.join(repo_root, 'PointPillars/model/layer'))
    sys.path.insert(0, os.path.join(repo_root, 'web_lidardetector'))
    from interface import ITrain

    cfg = load_config(config_path)
    train_cfg = cfg['TRAIN_MODEL']
    preprocess_cfg = cfg['PREPROCESS']
    dataset_dir = os.path.join(args.data_root, preprocess_cfg['SAVE_DATA_PATH'][0])
    required_files = ['training.pkl', 'validation.pkl']
    missing_files = [name for name in required_files if not os.path.exists(os.path.join(dataset_dir, name))]
    if missing_files:
        raise FileNotFoundError(
            f'Missing preprocessing outputs in {dataset_dir}: {missing_files}. '
            f'Run scripts/run_hx_preprocess.py --mode train first.'
        )

    if args.smoke and args.resume_epoch is not None:
        raise ValueError('--smoke cannot be used with --resume-epoch')

    if args.smoke:
        train_cfg['TRAIN']['CTRL']['CTRL_']['EPOCH_NUM'][0] = 1
        train_cfg['TRAIN']['CTRL']['DATA']['BATCH_SIZE'][0] = 1
        train_cfg['TRAIN']['OVERALL']['INITIAL_RESULT'][0] = True
        train_cfg['TRAIN']['CTRL']['LOSS']['PRINT_GAP'][0] = 1

    if args.epochs is not None:
        train_cfg['TRAIN']['CTRL']['CTRL_']['EPOCH_NUM'][0] = args.epochs
    if args.batch_size is not None:
        train_cfg['TRAIN']['CTRL']['DATA']['BATCH_SIZE'][0] = args.batch_size
    if args.eval_every_epochs is not None:
        train_cfg['TRAIN']['CTRL']['CTRL_']['EVAL_EVERY_EPOCHS'][0] = args.eval_every_epochs
    if args.num_workers is not None:
        train_cfg['TRAIN']['CTRL']['DATA']['NUM_WORKERS'][0] = args.num_workers
    if args.multi_gpu:
        train_cfg['TRAIN']['CTRL']['CTRL_']['USE_MULTI_GPU'][0] = True
    if args.resume_epoch is not None:
        if args.resume_epoch < 0:
            raise ValueError('--resume-epoch must be greater than or equal to 0')
        total_epochs = train_cfg['TRAIN']['CTRL']['CTRL_']['EPOCH_NUM'][0]
        if args.resume_epoch >= total_epochs:
            raise ValueError(
                f'--resume-epoch ({args.resume_epoch}) must be smaller than total epochs ({total_epochs}). '
                'Increase --epochs to continue training.'
            )
        train_cfg['TRAIN']['OVERALL']['INITIAL_RESULT'][0] = False
        train_cfg['TRAIN']['CTRL']['CTRL_']['CONTINUE_EPOCH'][0] = args.resume_epoch
        continue_prefix = train_cfg['TRAIN']['PATH']['CONTINUE_MODEL_PREFIX'][0]
        checkpoint_path = os.path.join(model_epoch_root, f'{continue_prefix}_{args.resume_epoch}.torch')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'Resume checkpoint not found: {checkpoint_path}')

    res = {'msg': []}
    ITrain(train_cfg, args.data_root, args.result_root, model_epoch_root, pretrained_path=None, check_flag=args.smoke, res_dict=res)

    for message in res['msg']:
        print(message)


if __name__ == '__main__':
    main()
