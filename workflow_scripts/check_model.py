import ort_test_dir_utils
import onnxruntime
import onnx
import test_utils


def by_onnx(model_path, model_name):
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    print('[PASS] {} is checked by onnx. '.format(model_name))

def by_onnxruntime(model_path, model_name, test_data_set):
    # if 'test_data_set_N' doesn't exist, create test_dir
    if not test_data_set:
        onnxruntime.InferenceSession(model_path)
        ort_test_dir_utils.create_test_dir(model_path, './', test_utils.TEST_ORT_DIR)
        ort_test_dir_utils.run_test_dir(test_utils.TEST_ORT_DIR)
    # otherwise use the existing 'test_data_set_N' as test data
    else:
        test_dir_from_tar = test_utils.get_model_directory(model_path)
        ort_test_dir_utils.run_test_dir(test_dir_from_tar)
    print('[PASS] {} is checked by onnxruntime. '.format(model_name))
    # remove the produced test_dir from ORT
    test_utils.remove_onnxruntime_test_dir()
