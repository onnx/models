from utils import ort_test_dir_utils
import onnxruntime
import onnx
from utils import test_utils


def run_onnx_checker(model):
    # stricter onnx.checker with onnx.shape_inference
    onnx.checker.check_model(model, True)

def run_backend_ort(model_path, test_data_set=None):
    # if 'test_data_set_N' doesn't exist, create test_dir
    if not test_data_set:
        onnxruntime.InferenceSession(model_path)
        ort_test_dir_utils.create_test_dir(model_path, './', test_utils.TEST_ORT_DIR)
        ort_test_dir_utils.run_test_dir(test_utils.TEST_ORT_DIR)
    # otherwise use the existing 'test_data_set_N' as test data
    else:
        test_dir_from_tar = test_utils.get_model_directory(model_path)
        ort_test_dir_utils.run_test_dir(test_dir_from_tar)
    # remove the produced test_dir from ORT
    test_utils.remove_onnxruntime_test_dir()
