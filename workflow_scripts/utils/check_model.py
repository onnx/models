# SPDX-License-Identifier: Apache-2.0

import onnx
from utils import ort_test_dir_utils, test_utils
import tarfile
import os.path

def run_onnx_checker(model_path):
    onnx.checker.check_model(model_path)

def run_backend_ort(model_path, test_data_set=None, tar_gz_path=None):
    model = onnx.load(model_path)
    if model.opset_import[0].version < 7:
        print('Skip ORT test since it only *guarantees* support for models stamped with opset version 7')
        return
    # if 'test_data_set_N' doesn't exist, create test_dir
    dir_path = model_path.split('/')[-1].replace('.onnx', '')
    if not test_data_set:
        ort_test_dir_utils.create_test_dir(model_path, './', dir_path)
        ort_test_dir_utils.run_test_dir(dir_path)
        os.remove(tar_gz_path)
        make_tarfile(tar_gz_path, dir_path)
    # otherwise use the existing 'test_data_set_N' as test data
    else:
        test_dir_from_tar = test_utils.get_model_directory(model_path)
        ort_test_dir_utils.run_test_dir(test_dir_from_tar)
    
    # remove the produced test_dir from ORT
    test_utils.remove_onnxruntime_test_dir()

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))