import onnx
from pathlib import Path
import subprocess
import sys
import check_model
import os
import test_utils


def main(argv):
  test_onnx, test_onnxruntime = True, True
  if len(argv) >= 2 and argv[1] == 'onnx':
    test_onnx, test_onnxruntime = True, False
  elif len(argv) >= 2 and argv[1] == 'onnxruntime':
    test_onnx, test_onnxruntime = False, True
    
  cwd_path = Path.cwd()
  # obtain list of added or modified files in this PR
  obtain_diff = subprocess.Popen(['git', 'diff', '--name-only', '--diff-filter=AM', 'origin/master', 'HEAD'],
  cwd=cwd_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdoutput, stderroutput = obtain_diff.communicate()
  diff_list = stdoutput.split()

  # identify list of changed onnx models in model Zoo
  model_list = [str(model).replace("b'","").replace("'", "") for model in diff_list if ".onnx" in str(model)]
  # run lfs install before starting the tests
  test_utils.run_lfs_install()

  print("\n=== Running ONNX Checker on added models ===\n")
  # run checker on each model
  failed_models = []
  for model_path in model_list:
      model_name = model_path.split('/')[-1]
      print('Testing {}'.format(model_name))

      try:
        # replace '.onnx' with '.tar.gz'
        tar_gz_path = model_path[::-5] + '.tar.gz'
        test_data = None
        # if tar.gz exists, use the onnx model 
        if os.path.exists(tar_gz_path):
          test_utils.pull_lfs_file(model_path)
          # check whether 'test_data_set_0' exists
          model_path_from_tar, test_data = test_utils.extract_test_data(tar_gz_path)
          # check the onnx model from .tar.gz
          if test_onnx: check_model.by_onnx(model_path_from_tar, model_name)
          if test_onnxruntime: check_model.by_onnxruntime(model_path_from_tar, model_name, test_data)
          print('Model {} from .tar.gz has been successfully checked. '.format(model_name))
        else:
          test_utils.pull_lfs_file(model_path)

        # check the onnx model from GitHub
        if test_onnx: check_model.by_onnx(model_path, model_name)
        if test_onnxruntime: check_model.by_onnxruntime(model_path, model_name, test_data)
        print('Model {} from GitHub has been successfully checked. '.format(model_name))

      except Exception as e:
        print(e)
        failed_models.append(model_path)
      test_utils.remove_test_dir()

  if len(failed_models) == 0:
      print('{} models have been checked.'.format(len(model_list)))
  else:
      print('In all {} models, {} models failed.'.format(len(model_list), len(failed_models)))
      sys.exit(1)

if __name__ == '__main__':
    main(sys.argv)