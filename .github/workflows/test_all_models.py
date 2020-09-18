import onnx
from pathlib import Path
import subprocess
import requests
import ort_test_dir_utils
import onnxruntime
import sys
import os
import shutil
import time

cwd_path = Path.cwd()
TEST_DIR = 'test_dir'

def run_lfs_install():
    result = subprocess.run(['git', 'lfs', 'install'], cwd=cwd_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print('Git LFS install completed with return code= {}'.format(result.returncode))


def pull_lfs_file(file_name):
    result = subprocess.run(['git', 'lfs', 'pull', '--include', file_name, '--exclude', '\'\''], cwd=cwd_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print('LFS pull completed with return code= {}'.format(result.returncode))


def save_file(url, file_name):
  r = requests.get(url)
  with open(file_name, 'wb') as f:
    f.write(r.content)

def check_by_onnx(model_path, model_name):
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    print('[PASS]: {} is checked by onnx. '.format(model_name))

def check_by_onnxruntime(model_path, model_name):
    onnxruntime.InferenceSession(model_path)
    ort_test_dir_utils.create_test_dir(model_path, './', TEST_DIR)
    ort_test_dir_utils.run_test_dir(TEST_DIR)
    print('[PASS]: {} is checked by onnxruntime. '.format(model_name))


def main():
    model_directory = ['text/machine_comprehension/t5/'] # ['text', 'vision'] # ['vision/classification/efficientnet-lite4/']
    model_list = []

    for directory in model_directory:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.onnx'):
                    model_list.append(os.path.join(root, file))
                    print(os.path.join(root, file))


    # run lfs install before starting the tests
    run_lfs_install()

    print('\n=== Running ONNX Checker on added models ===\n')
    # run checker on each model
    failed_models = []
    for model_path in model_list:
        start = time.time()
        model_name = model_path.split('/')[-1]
        print('----------------Testing: {}----------------'.format(model_name))

        try:
            pull_lfs_file(model_path)
            end = time.time()
            print('--------------Time used: {} secs--------------'.format(end - start))
            check_by_onnx(model_path, model_name)
            check_by_onnxruntime(model_path, model_name)
            if os.path.exists(TEST_DIR) and os.path.isdir(TEST_DIR):
                shutil.rmtree(TEST_DIR)

        except Exception as e:
            print('[FAIL]: {}'.format(e))
            failed_models.append(model_path)



    if len(failed_models) == 0:
        print('{} models have been checked.'.format(len(model_list)))
    else:
        print('In all {} models, {} models failed.'.format(len(model_list), len(failed_models)))
        sys.exit(1)
      

if __name__ == '__main__':
    main()