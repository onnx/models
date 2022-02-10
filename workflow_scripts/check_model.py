# SPDX-License-Identifier: Apache-2.0

import ort_test_dir_utils
import onnxruntime
import onnx
import test_utils


def run_onnx_checker(model_path):
    model = onnx.load(model_path)
    onnx.checker.check_model(model)

def run_backend_ort(model_path, test_data_set=None):
    model = onnx.load(model_path)
    if model.opset_import[0].version < 7:
        print('Skip ORT test since it only *guarantees* support for models stamped with opset version 7')
        return
    # if 'test_data_set_N' doesn't exist, create test_dir
    if not test_data_set:
        # Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
        # other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
        # based on the build flags) when instantiating InferenceSession.
        # For example, if NVIDIA GPU is available and ORT Python package is built with CUDA, then call API as following:
        # onnxruntime.InferenceSession(path/to/model, providers=['CUDAExecutionProvider'])
        onnxruntime.InferenceSession(model_path)
        ort_test_dir_utils.create_test_dir(model_path, './', test_utils.TEST_ORT_DIR)
        ort_test_dir_utils.run_test_dir(test_utils.TEST_ORT_DIR)
    # otherwise use the existing 'test_data_set_N' as test data
    else:
        test_dir_from_tar = test_utils.get_model_directory(model_path)
        ort_test_dir_utils.run_test_dir(test_dir_from_tar)
    # remove the produced test_dir from ORT
    test_utils.remove_onnxruntime_test_dir()
