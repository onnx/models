import os
import subprocess
import sys
from pathlib import Path

python_root = "models/python"
cache_dir = ".cache"
cwd_path = Path.cwd()
errors = 0
total_models = 0
ZOO_OPSET_VERSION = "18"

def find_base_onnx(root_dir):
    for root, _, files in os.walk(root_dir):
        for file in files:
            if "-base.onnx" in file:
                return os.path.join(root, file)
    return None

for root, dirs, files in os.walk(python_root):
    for dir in dirs:
        dir_path = os.path.join(root, dir)
        model_name = dir + ".onnx"
        model_path = os.path.join(dir_path, model_name)
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
        subprocess.run(["pip", "install", "-r", requirements_path], cwd=cwd_path, stdout=sys.stdout,
                        stderr=sys.stderr)
        os.remove(model_path)
        subprocess.run(["python", model_python_path], cwd=cwd_path, stdout=sys.stdout,
                        stderr=sys.stderr)
        if not os.path.exists(model_name):
            print(f"Model {model_path} was not created by {model_python_path}.")
            errors += 1
            continue
        os.replace(model_name, model_path)
        subprocess.run(["git", "diff", "--exit-code", "--", model_path],
                        cwd=cwd_path, stdout=sys.stdout,
                        stderr=sys.stderr)
        os.remove(model_path)
        subprocess.run(["benchit", model_python_path, "--cache-dir", cache_dir, "--onnx-opset", ZOO_OPSET_VERSION],
                        cwd=cwd_path, stdout=subprocess.PIPE,
                        stderr=sys.stderr)
        cache_model = find_base_onnx(cache_dir)
        if cache_dir is None:
            print(f"Model {model_path} was not created by benchit from mlagility.")
            errors += 1
            continue
        os.replace(cache_model, model_path)
        subprocess.run(["git", "diff", "--exit-code", "--", model_path],
                        cwd=cwd_path, stdout=sys.stdout,
                        stderr=sys.stderr)

        print(f"Successfully checked {model_path}.")


if errors > 0:
    print(f"All {total_models} model(s) have been checked, but {errors} model(s) failed.")
    sys.exit(1)
else:
    print(f"All {total_models} model(s) have been checked.")