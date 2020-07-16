import onnx
from pathlib import Path
import subprocess
import requests

def save_file(url, file_name):
  r = requests.get(url)
  with open(file_name, 'wb') as f:
    f.write(r.content)

PIPE = subprocess.PIPE
cwd_path = Path.cwd()

# obtain list of added or modified files in this PR
obtain_diff = subprocess.Popen(['git', 'diff', '--name-only', '--diff-filter=AM', 'origin/master', 'HEAD'], cwd=cwd_path, stdout=PIPE, stderr=PIPE)
stdoutput, stderroutput = obtain_diff.communicate()
diff_list = stdoutput.split()

# obtain git commit ID to extract git lfs models accordingly
obtain_commit_id = subprocess.Popen(['git', 'rev-parse', 'HEAD'], stdout=PIPE, stderr=PIPE)
stdoutput, stderroutput = obtain_commit_id.communicate()
commit_id = str(stdoutput).replace("b'","").replace("'", "")[:-2]

# identify list of changed onnx models in model Zoo
model_list = [str(model).replace("b'","").replace("'", "") for model in diff_list if ".onnx" in str(model)]

print("\n=== Running ONNX Checker on added models ===\n")
# run checker on each model
for model_path in model_list:
    model_name = model_path.split('/')[-1]
    print("Testing:", model_name)

    model_url = "https://github.com/onnx/models/raw/" + commit_id + "/" + model_path
    save_file(model_url, model_path)

    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    print("Model", model_name, "has been successfully checked!")

print(len(model_list), "model(s) checked.")
