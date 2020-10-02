import argparse
import check_model
import os
from pathlib import Path
import subprocess
import sys
import test_utils

def check_by_target(model_path, file_name, test_data_set, target):
  if target != 'onnxruntime':
    check_model.run_onnx_checker(model_path, file_name)
  if target != 'onnx':
    check_model.run_backend_ort(model_path, file_name, test_data_set)

def main():
  parser = argparse.ArgumentParser(description='Test settings')
  # default test by both onnx and onnxruntime
  # if target is specified, only test by the specified one
  parser.add_argument('--target', help='Test the model by which (onnx/onnxruntime)?')
  args = parser.parse_args()
    
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

  print('\n=== Running ONNX Checker on added models ===\n')
  # run checker on each model
  failed_models = []
  tar_ext_name = '.tar.gz'
  for model_path in model_list:
      model_name = model_path.split('/')[-1]
      tar_name = model_name.replace('.onnx', tar_ext_name)
      print('==============Testing {}=============='.format(model_name))

      try:
        # replace '.onnx' with '.tar.gz'
        tar_gz_path = model_path[:-5] + '.tar.gz'
        test_data_set = []
        # if tar.gz exists, git pull and try to get test data
        if os.path.exists(tar_gz_path):
          test_utils.pull_lfs_file(tar_gz_path)
          # check whether 'test_data_set_0' exists
          model_path_from_tar, test_data_set = test_utils.extract_test_data(tar_gz_path)

          # 1. check the onnx model from .tar.gz
          check_by_target(model_path_from_tar, tar_name, test_data_set, args.target)
        
        # git pull the onnx file
        test_utils.pull_lfs_file(model_path)

        # 2. check the uploaded onnx model
        check_by_target(model_path, model_name, test_data_set, args.target)
        
        if os.path.exists(tar_gz_path):
          print('[SUCCESS] Both {} and {} checked. '.format(tar_name, model_name))
        else:
          print('[SUCCESS] {} checked. '.format(model_name))

      except Exception as e:
        print('[FAIL] {}: {}'.format(model_name, e))
        failed_models.append(model_path)
        test_utils.remove_onnxruntime_test_dir()

      # remove the produced tar directory
      test_utils.remove_tar_dir()

  if len(failed_models) == 0:
      print('{} models have been checked. '.format(len(model_list)))
  else:
      print('In all {} models, {} models failed. '.format(len(model_list), len(failed_models)))
      sys.exit(1)

if __name__ == '__main__':
    main()