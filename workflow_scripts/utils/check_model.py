# SPDX-License-Identifier: Apache-2.0

import onnx
from utils import ort_test_dir_utils, test_utils

def run_onnx_checker(model_path):
    onnx.checker.check_model(model_path)

def run_backend_ort(model_path, test_data_set=None):
    model = onnx.load(model_path)
    if model.opset_import[0].version < 7:
        print('Skip ORT test since it only *guarantees* support for models stamped with opset version 7')
        return
    # if 'test_data_set_N' doesn't exist, create test_dir
    if not test_data_set:
        ort_test_dir_utils.create_test_dir(model_path, './', test_utils.TEST_ORT_DIR)
        ort_test_dir_utils.run_test_dir(test_utils.TEST_ORT_DIR)
    # otherwise use the existing 'test_data_set_N' as test data
    else:
        test_dir_from_tar = test_utils.get_model_directory(model_path)
        ort_test_dir_utils.run_test_dir(test_dir_from_tar)
    # remove the produced test_dir from ORT
    test_utils.remove_onnxruntime_test_dir()