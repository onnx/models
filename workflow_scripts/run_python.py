import os
import subprocess
import sys
from pathlib import Path

python_root = "models/python"
cwd_path = Path.cwd()
errors = 0
total_models = 0

for root, dirs, files in os.walk(python_root):
    for dir in dirs:
        dir_path = os.path.join(root, dir)
        model_path = os.path.join(dir_path, dir + ".onnx")
        requirements_path = os.path.join(dir_path, "requirements.txt")
        model_python_path = os.path.join(dir_path, "model.py")
        if not os.path.exists(model_python_path) and not os.path.exists(model_path):
            continue
        total_models += 1
        if not os.path.exists(model_python_path):
            print(f"Model {model_python_path} does not exist.")
            errors += 1
            continue
        if not os.path.exists(model_path):
            print(f"Model {model_path} does not exist.")
            errors += 1
            continue
        if not os.path.exists(requirements_path):
            errors += 1
            continue
        os.remove(model_path)
        cwd_path = dir_path
        subprocess.run(["pip", "install", "-r", requirements_path], cwd=cwd_path, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
        subprocess.run(["python", model_python_path], cwd=cwd_path, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
        if not os.path.exists(model_path):
            print(f"Model {model_path} was not created by {model_python_path}.")
            errors += 1
            continue
        subprocess.run(["git", "diff", "--exit-code", "--", model_path],
                        cwd=cwd_path, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
        print(f"Successfully checked {model_path}.")


if errors > 0:
    print(f"All {total_models} models have been checked, but {errors} model(s) failed.")
    sys.exit(1)
else:
    print(f"All {total_models} models have been checked.")