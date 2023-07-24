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
    print(f'Git LFS install completed with return code= {result.returncode}')


def pull_lfs_file(file_name):
    result = subprocess.run(['git', 'lfs', 'pull', '--include', file_name, '--exclude', '\'\''], cwd=cwd_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f'LFS pull completed for {file_name} with return code= {result.returncode}')


def pull_lfs_directory(directory_name):
    # git lfs pull those test_data_set_* folders
    for _, dirs, _ in os.walk(directory_name):
        for dir in dirs:
            if "test_data_set_" in dir:
                test_data_set_dir = os.path.join(directory_name, dir)
                for _, _, files in os.walk(test_data_set_dir):
                    for file in files:
                        if file.endswith(".pb"):
                            pull_lfs_file(os.path.join(test_data_set_dir, file))


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


def get_changed_models():
    tar_ext_name = ".tar.gz"
    onnx_ext_name = ".onnx"
    model_list = []
    cwd_path = Path.cwd()
    # TODO: use the main branch instead of new-models
    branch_name = "new-models" # "main"
    # git fetch first for git diff on GitHub Action
    subprocess.run(["git", "fetch", "origin", f"{branch_name}:{branch_name}"],
                   cwd=cwd_path, stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE)
    # obtain list of added or modified files in this PR
    obtain_diff = subprocess.Popen(["git", "diff", "--name-only", "--diff-filter=AM", "origin/" + branch_name, "HEAD"],
                                   cwd=cwd_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdoutput, _ = obtain_diff.communicate()
    diff_list = stdoutput.split()

    # identify list of changed ONNX models in ONXX Model Zoo
    model_list = [str(model).replace("b'", "").replace("'", "")
                  for model in diff_list if onnx_ext_name in str(model) or tar_ext_name in str(model)]
    return model_list
