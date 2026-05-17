import argparse
import os
import pickle
import random
import tempfile
from typing import Any


def load_infos(file_path: str) -> list[dict[str, Any]]:
    with open(file_path, 'rb') as file:
        infos = pickle.load(file)
    if not isinstance(infos, list):
        raise ValueError(f'Expected list in {file_path}, got {type(infos).__name__}')
    return infos


def dump_infos(file_path: str, infos: list[dict[str, Any]]) -> None:
    output_dir = os.path.dirname(file_path) or '.'
    temp_path = ''
    try:
        with tempfile.NamedTemporaryFile('wb', dir=output_dir, delete=False) as file:
            temp_path = file.name
            pickle.dump(infos, file)
        os.replace(temp_path, file_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def select_subset(infos: list[dict[str, Any]], limit: int, seed: int, shuffle: bool) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError('--limit must be greater than 0')
    if len(infos) <= limit:
        return list(infos)
    if not shuffle:
        return list(infos[:limit])

    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(infos)), limit))
    return [infos[index] for index in indices]


def resolve_child_path(parent_dir: str, child_name: str) -> str:
    if os.path.basename(child_name) != child_name:
        raise ValueError(f'Expected a file name without path segments, got: {child_name}')
    return os.path.join(parent_dir, child_name)


def main() -> None:
    parser = argparse.ArgumentParser(description='Create a small HX debug split from existing preprocessing pkl files.')
    parser.add_argument('--input-dir', required=True, help='Directory containing training.pkl/validation.pkl from preprocessing output')
    parser.add_argument('--output-dir', required=True, help='Directory to write the debug pkl files')
    parser.add_argument('--train-limit', type=int, default=64, help='Max number of training samples to keep')
    parser.add_argument('--validation-limit', type=int, default=16, help='Max number of validation samples to keep')
    parser.add_argument('--train-file', default='training.pkl', help='Input/output training pkl name')
    parser.add_argument('--validation-file', default='validation.pkl', help='Input/output validation pkl name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed used when --shuffle is enabled')
    parser.add_argument('--shuffle', action='store_true', help='Randomly sample instead of taking the first N samples')
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    train_input_path = resolve_child_path(input_dir, args.train_file)
    validation_input_path = resolve_child_path(input_dir, args.validation_file)
    train_output_path = resolve_child_path(output_dir, args.train_file)
    validation_output_path = resolve_child_path(output_dir, args.validation_file)

    normalized_input_dir = os.path.normcase(os.path.normpath(input_dir))
    normalized_output_dir = os.path.normcase(os.path.normpath(output_dir))
    if normalized_input_dir == normalized_output_dir:
        raise ValueError('--output-dir must be different from --input-dir to avoid overwriting the source split files')
    if train_input_path == validation_input_path:
        raise ValueError('--train-file and --validation-file must be different input files')
    if train_output_path == validation_output_path:
        raise ValueError('--train-file and --validation-file must be different output files')

    if not os.path.exists(train_input_path):
        raise FileNotFoundError(f'Missing input file: {train_input_path}')
    if not os.path.exists(validation_input_path):
        raise FileNotFoundError(f'Missing input file: {validation_input_path}')

    os.makedirs(output_dir, exist_ok=True)

    train_infos = load_infos(train_input_path)
    validation_infos = load_infos(validation_input_path)

    debug_train_infos = select_subset(train_infos, args.train_limit, args.seed, args.shuffle)
    debug_validation_infos = select_subset(validation_infos, args.validation_limit, args.seed + 1, args.shuffle)

    dump_infos(train_output_path, debug_train_infos)
    dump_infos(validation_output_path, debug_validation_infos)

    print(f'Train: {len(debug_train_infos)}/{len(train_infos)} -> {train_output_path}')
    print(f'Validation: {len(debug_validation_infos)}/{len(validation_infos)} -> {validation_output_path}')


if __name__ == '__main__':
    main()
