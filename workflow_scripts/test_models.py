# SPDX-License-Identifier: Apache-2.0

import argparse
import check_model
from pathlib import Path
import subprocess
import sys
import test_utils
import os


def get_all_models():
    model_list = []
    parent_dir = ["text", "vision"]
    for directory in parent_dir:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.tar.gz') or file.endswith('.onnx'):
                    onnx_model_path = os.path.join(root, file)
                    model_list.append(onnx_model_path)
    return model_list


def main():
    parser = argparse.ArgumentParser(description='Test settings')
    # default all: test by both onnx and onnxruntime
    # if target is specified, only test by the specified one
    parser.add_argument('--target', required=False, default='all', type=str,
                        help='Test the model by which (onnx/onnxruntime)?',
                        choices=['onnx', 'onnxruntime', 'all'])
    # set it True to update broken test_data_set
    create_if_failed = False
    args = parser.parse_args()

    cwd_path = Path.cwd()
    # git fetch first for git diff on GitHub Action
    subprocess.run(['git', 'fetch', 'origin', 'main:main'],
                   cwd=cwd_path, stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE)
    # obtain list of added or modified files in this PR
    obtain_diff = subprocess.Popen(['git', 'diff', '--name-only', '--diff-filter=AM', 'origin/main', 'HEAD'],
                                   cwd=cwd_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdoutput, _ = obtain_diff.communicate()
    diff_list = stdoutput.split()

    # identify list of changed ONNX models in ONXX Model Zoo
    tar_ext_name = '.tar.gz'
    onnx_ext_name = '.onnx'
    model_list = [str(model).replace("b'", "").replace("'", "")
                  for model in diff_list if onnx_ext_name in str(model) or tar_ext_name in str(model)]
    # run lfs install before starting the tests
    test_utils.run_lfs_install()

    print('\n=== Running ONNX Checker on added models ===\n')
    # run checker on each model
    failed_models = []
    for model_path in model_list:
        model_name = model_path.split('/')[-1]
        print('==============Testing {}=============='.format(model_name))

        try:
            # check .tar.gz by ORT and ONNX
            if tar_ext_name in model_name:
                # Step 1: check the ONNX model and test_data_set from .tar.gz by ORT
                test_data_set = []
                test_utils.pull_lfs_file(model_path)
                # check whether 'test_data_set_0' exists
                model_path_from_tar, test_data_set = test_utils.extract_test_data(model_path)
                # if tar.gz exists, git pull and try to get test data
                if (args.target == 'onnxruntime' or args.target == 'all'):
                    # finally check the ONNX model from .tar.gz by ORT
                    # if the test_data_set does not exist, create the test_data_set
                    try:
                        check_model.run_backend_ort(model_path_from_tar, test_data_set)
                        print('[PASS] {} is checked by onnxruntime. '.format(model_name))
                    except Exception as e:
                        if not create_if_failed:
                            raise Exception(e)
                        else:
                            print('Warning: original test data for {} is broken: {}'.format(model_path, e))
                        if '-int8' not in model_name:
                            check_model.run_backend_ort(model_path_from_tar, None, model_name)
                        else:
                            print('Skip int8 models because their test_data_set was created in avx512_vnni machines')
                        print('[PASS] {} is checked by onnxruntime. '.format(model_name))
                # Step 2: check the ONNX model inside .tar.gz by ONNX
                if args.target == 'onnx' or args.target == 'all':
                    check_model.run_onnx_checker(model_path_from_tar)
                    print('[PASS] {} is checked by onnx. '.format(model_name))
            # check uploaded standalone ONNX model by ONNX
            elif onnx_ext_name in model_name:
                if args.target == 'onnx' or args.target == 'all':
                    test_utils.pull_lfs_file(model_path)
                    check_model.run_onnx_checker(model_path)
                    print('[PASS] {} is checked by onnx. '.format(model_name))

        except Exception as e:
            print('[FAIL] {}: {}'.format(model_name, e))
            failed_models.append(model_path)

        # remove checked models and directories to save space in CIs
        if os.path.exists(model_path):
            os.remove(model_path)
        test_utils.remove_onnxruntime_test_dir()
        test_utils.remove_tar_dir()
        test_utils.run_lfs_prune()

    if len(failed_models) == 0:
        print('{} models have been checked. '.format(len(model_list)))
    else:
        print('In all {} models, {} models failed. '.format(len(model_list), len(failed_models)))
        sys.exit(1)


if __name__ == '__main__':
    main()
