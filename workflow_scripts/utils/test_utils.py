from pathlib import Path
import subprocess
import tarfile
import os
import shutil
from utils import check_model
import time
import sys
import onnx

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
    print('LFS prune completed with return code= {}'.format(result.returncode))

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
                onnx_model = os.path.join(root, file)
            for subdir in dirs:
            # detect any test_data_set
                if subdir.startswith('test_data_set_'):
                    subdir_path = os.path.join(root, subdir)
                    test_data_set.append(subdir_path)
    return onnx_model, test_data_set

def remove_tar_dir():
    if os.path.exists(TEST_TAR_DIR) and os.path.isdir(TEST_TAR_DIR):
        shutil.rmtree(TEST_TAR_DIR)


def remove_onnxruntime_test_dir():
    if os.path.exists(TEST_ORT_DIR) and os.path.isdir(TEST_ORT_DIR):
        shutil.rmtree(TEST_ORT_DIR)
     

def test_models(model_list, target, skip_list=[]):
    """
    model_list: a list of model path which will be tested
    target: all, onnx, onnxruntime 
    Given model_list, pull and test them by target
    including model check and test data validation
    eventually remove all files to save space in CIs
    """
    # run lfs install before starting the tests
    run_lfs_install()
    failed_models = []
    skip_models = []
    tar_ext_name = '.tar.gz'
    for model_path in model_list:
        start = time.time()
        model_name = model_path.split('/')[-1]
        tar_name = model_name.replace('.onnx', tar_ext_name)
        print('==============Testing {}=============='.format(model_name))
        tar_gz_path = model_path[:-5] + '.tar.gz'
        test_data_set = []
        try:
            # Step 1: check the uploaded onnx model by ONNX
            # git pull the onnx file
            pull_lfs_file(model_path)
            if target == 'onnx' or target == 'all':
                if model_path in skip_list:
                    skip_models.append(model_name)
                    print('SKIP {} is in the skip list for checker. '.format(model_name))
                    continue

                model = onnx.load(model_path)
                # check original model
                check_model.run_onnx_checker(model)
                # check inferred model as well
                inferred_model = onnx.shape_inference.infer_shapes(model)
                check_model.run_onnx_checker(inferred_model)
                print('[PASS] {} is checked by onnx. '.format(model_name))

            # Step 2: check the onnx model and test_data_set from .tar.gz by ORT
            # if tar.gz exists, git pull and try to get test data
            if (target == 'onnxruntime' or target == 'all') and os.path.exists(tar_gz_path):
                pull_lfs_file(tar_gz_path)
                # check whether 'test_data_set_0' exists
                model_path_from_tar, test_data_set = extract_test_data(tar_gz_path)
                # finally check the onnx model from .tar.gz by ORT
                # if the test_data_set does not exist, create the test_data_set
                try:
                    check_model.run_backend_ort(model_path_from_tar, test_data_set)
                except:
                    print('Warning: original test data for {} is broken. '.format(model_path))
                    check_model.run_backend_ort(model_path_from_tar, None)
                print('[PASS] {} is checked by onnxruntime. '.format(tar_name))

            end = time.time()
            print('--------------Time used: {} secs-------------'.format(end - start))

        except Exception as e:
            print('[FAIL] {}: {}'.format(model_name, e))
            failed_models.append(model_path)
        
        # remove the model/tar files to save space in CIs
        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists(tar_gz_path):
            os.remove(tar_gz_path)
        # remove the produced tar/test directories
        remove_tar_dir()
        remove_onnxruntime_test_dir()
        # clean git lfs cache
        run_lfs_prune()

    if len(failed_models) == 0:
        print('{} models have been checked. '.format(len(model_list)))
    else:
        print('In all {} models, {} models failed, {} models were skipped. '.format(len(model_list), len(failed_models), len(skip_models)))
        sys.exit(1)
