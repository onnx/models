import onnx
from pathlib import Path
import subprocess
import sys

def run_lfs_install():
  result = subprocess.run(['git', 'lfs', 'install'], cwd=cwd_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  print("Git LFS install completed with return code=" + str(result.returncode))

def pull_lfs_file(file_name):
  result = subprocess.run(['git', 'lfs', 'pull', '--include', file_name, '--exclude', '\"\"'], cwd=cwd_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  print("LFS pull completed with return code=" + str(result.returncode))

cwd_path = Path.cwd()

# obtain list of added or modified files in this PR
obtain_diff = subprocess.Popen(['git', 'diff', '--name-only', '--diff-filter=AM', 'origin/master', 'HEAD'],
 cwd=cwd_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdoutput, stderroutput = obtain_diff.communicate()
diff_list = stdoutput.split()

# identify list of changed onnx models in model Zoo
model_list = [str(model).replace("b'","").replace("'", "") for model in diff_list if ".onnx" in str(model)]

# run lfs install before starting the tests
run_lfs_install()

print("\n=== Running ONNX Checker on added models ===\n")
# run checker on each model
failed_models = []
for model_path in model_list:
    model_name = model_path.split('/')[-1]
    print("Testing:", model_name)

    try:
      pull_lfs_file(model_path)
      model = onnx.load(model_path)
      onnx.checker.check_model(model)
      print("Model", model_name, "has been successfully checked!")
    except Exception as e:
      print(e)
      failed_models.append(model_path)

if len(failed_models) != 0:
  print(str(len(failed_models)) +" models failed onnx checker.")
  sys.exit(1)

print(len(model_list), "model(s) checked.")
