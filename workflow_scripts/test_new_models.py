import argparse
from pathlib import Path
import subprocess
from utils import test_utils


def main():
  parser = argparse.ArgumentParser(description='Test settings')
  # default all: test by both onnx and onnxruntime
  # if target is specified, only test by the specified one
  parser.add_argument('--target', required=False, default='all', type=str, 
                      help='Test the model by which (onnx/onnxruntime)?',
                      choices=['onnx', 'onnxruntime', 'all'])
  args = parser.parse_args()

  cwd_path = Path.cwd()
  # git fetch first for git diff on GitHub Action
  subprocess.run(['git', 'fetch', 'origin', 'master:master'], cwd=cwd_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  # obtain list of added or modified files in this PR
  obtain_diff = subprocess.Popen(['git', 'diff', '--name-only', '--diff-filter=AM', 'origin/master', 'HEAD'],
  cwd=cwd_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdoutput, stderroutput = obtain_diff.communicate()
  diff_list = stdoutput.split()

  # identify list of changed onnx models in model Zoo
  model_list = [str(model).replace("b'","").replace("'", "") for model in diff_list if ".onnx" in str(model)]

  print('=== Running Test on added {} models ==='.format(len(model_list)))
  test_utils.test_models(model_list, args.target)


if __name__ == '__main__':
    main()