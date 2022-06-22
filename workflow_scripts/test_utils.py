# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import subprocess
import tarfile
import os
from shutil import rmtree

TEST_ORT_DIR = 'ci_test_dir'
TEST_TAR_DIR = 'ci_test_tar_dir'
cwd_path = Path.cwd()


def get_model_directory(model_path):
    return os.path.dirname(model_path)


def run_lfs_install():
    result = subprocess.run(['git', 'lfs', 'install'], cwd=cwd_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print('Git LFS install completed with return code= {}'.format(result.returncode))


def pull_lfs_file(file_name):
    result = subprocess.run(['git', 'lfs', 'pull', '--include', file_name, '--exclude', '\'\''], cwd=cwd_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print('LFS pull completed with return code= {}'.format(result.returncode))


def run_lfs_prune():
    result = subprocess.run(['git', 'lfs', 'prune'], cwd=cwd_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f'LFS prune completed with return code= {result.returncode}')


def extract_test_data(file_path):
    tar = tarfile.open(file_path, "r:gz")
    tar.extractall(TEST_TAR_DIR)
    tar.close()
    return get_model_and_test_data(TEST_TAR_DIR)


def get_model_and_test_data(directory_path):
    onnx_model = None
    test_data_set = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.onnx'):
                file_path = os.path.join(root, file)
                assert onnx_model is None, "More than one ONNX model detected"
                onnx_model = file_path
        for subdir in dirs:
            # detect any test_data_set
            if subdir.startswith('test_data_set_'):
                subdir_path = os.path.join(root, subdir)
                test_data_set.append(subdir_path)
    return onnx_model, test_data_set


def remove_tar_dir():
    if os.path.exists(TEST_TAR_DIR) and os.path.isdir(TEST_TAR_DIR):
        rmtree(TEST_TAR_DIR)


def remove_onnxruntime_test_dir():
    if os.path.exists(TEST_ORT_DIR) and os.path.isdir(TEST_ORT_DIR):
        rmtree(TEST_ORT_DIR)
