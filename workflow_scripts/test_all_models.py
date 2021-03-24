# SPDX-License-Identifier: Apache-2.0

import argparse
import config
import os
from pathlib import Path
import subprocess
import sys
from utils import test_utils


def main():
    parser = argparse.ArgumentParser(description='Test settings')
    # default: test all models in the repo
    # if test_dir is specified, only test files under that specified path
    parser.add_argument('--test_dir', required=False, default='', type=str, 
                        help='Directory path for testing. e.g., text, vision')
    # default all: test by both onnx and onnxruntime
    # if target is specified, only test by the specified one
    parser.add_argument('--target', required=False, default='all', type=str, 
                        help='Test the model by which (onnx/onnxruntime)?',
                        choices=['onnx', 'onnxruntime', 'all'])                        
    args = parser.parse_args()
    parent_dir = []

    # collect the paths of models
    # if not set, go throught each directory
    if not args.test_dir:
        for file in os.listdir():
            if os.path.isdir(file):
                parent_dir.append(file)
    else:
        parent_dir.append(args.test_dir)
    model_list = []
    for directory in parent_dir:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.onnx'):
                    onnx_model_path = os.path.join(root, file)
                    model_list.append(onnx_model_path)
                    print(onnx_model_path)
    print('=== Running Test on all {} models ==='.format(len(model_list)))
    test_utils.test_models(model_list, args.target, True,
        config.SKIP_CHECKER_MODELS, config.SKIP_ORT_MODELS)


if __name__ == '__main__':
    main()