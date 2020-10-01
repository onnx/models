from pathlib import Path
import subprocess
import tarfile
import os
import shutil

TEST_ORT_DIR = 'ci_test_dir'
TEST_TAR_DIR = 'ci_test_tar_dir'
cwd_path = Path.cwd()

def get_model_directory(model_path):
    return '/'.join(model_path.split('/')[:-1])

def run_lfs_install():
    result = subprocess.run(['git', 'lfs', 'install'], cwd=cwd_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print('Git LFS install completed with return code= {}'.format(result.returncode))

def pull_lfs_file(file_name):
    result = subprocess.run(['git', 'lfs', 'pull', '--include', file_name, '--exclude', '\'\''], cwd=cwd_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print('LFS pull completed with return code= {}'.format(result.returncode))

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
            file_path = os.path.join(root, file)
            if file.endswith('.onnx'):
                onnx_model = file_path
            # detect any test_data_set
            elif file.startswith('test_data_set_'):
                test_data_set.append(file_path)
    return onnx_model, test_data_set

def remove_tar_dir():
    if os.path.exists(TEST_TAR_DIR) and os.path.isdir(TEST_TAR_DIR):
        shutil.rmtree(TEST_TAR_DIR)

def remove_onnxruntime_test_dir():
    if os.path.exists(TEST_ORT_DIR) and os.path.isdir(TEST_ORT_DIR):
        shutil.rmtree(TEST_ORT_DIR)        
