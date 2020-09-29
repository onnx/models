import ort_test_dir_utils
import onnxruntime
import onnx
import os
import test_utils
import shutil


def by_onnx(model_path, model_name):
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    print('[PASS]: {} is checked by onnx. '.format(model_name))

def by_onnxruntime(model_path, model_name, test_data_set_0):
    # if 'test_data_set_0' doesn't exist, create test_dir
    if test_data_set_0 is None:
        onnxruntime.InferenceSession(model_path)
        ort_test_dir_utils.create_test_dir(model_path, './', test_utils.TEST_DIR)
        ort_test_dir_utils.run_test_dir(test_utils.TEST_DIR)
    else:
        test_dir = test_utils.get_model_directory(model_path)
        # copy est_data_set_0 to test_dir for run_test_dir
        shutil.copytree(test_data_set_0, test_dir)
        ort_test_dir_utils.run_test_dir(test_dir)
    print('[PASS]: {} is checked by onnxruntime. '.format(model_name))
}