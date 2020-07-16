import onnx
from pathlib import Path
import subprocess

PIPE = subprocess.PIPE
cwd_path = Path.cwd()
obtain_diff = subprocess.Popen(['git', 'diff', '--name-only', '--diff-filter=AM', 'origin/master', 'HEAD'], cwd=cwd_path, stdout=PIPE, stderr=PIPE)

stdoutput, stderroutput = obtain_diff.communicate()
diff_list = stdoutput.split()

# identify list of changed onnx models in model Zoo
model_list = [str(model).replace("b'","").replace("'", "") for model in diff_list if ".onnx" in str(model)]

# run checker on each model
for model_path in model_list:
    model_name = model_path.split('/')[-2]
    print("Testing ", model_name, ".")
    pull_model = subprocess.Popen(['git', 'lfs', 'pull', '--include=', model_path], cwd=cwd_path)
    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    print("Model ", model_name, "has been successfully checked!")

print(len(model_list), " models checked.")
