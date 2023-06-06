import config
import os.path as osp
from pathlib import Path
import shutil
import subprocess
import sys


base_name = "-op18-base.onnx"
cwd_path = Path.cwd()
mlagility_root = "mlagility/models"
mlagility_models_dir = "models/mlagility"
ZOO_OPSET_VERSION = "18"

errors = 0

for script_path, model_name, model_zoo_path in config.models_info:
    try:
        final_model_path = osp.join(mlagility_models_dir, model_zoo_path.replace(".onnx", "-" + ZOO_OPSET_VERSION + ".onnx"))
        subprocess.run(["benchit", osp.join(mlagility_root, script_path), "--cache-dir", mlagility_models_dir, "--onnx-opset", ZOO_OPSET_VERSION],
                        cwd=cwd_path, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
        shutil.copy(osp.join(mlagility_models_dir, model_name, "onnx", model_name + base_name), final_model_path)
        subprocess.run(["git", "diff", "--exit-code", "--", final_model_path],
                        cwd=cwd_path, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
        print(f"Successfully checked {model_zoo_path}.")
    except Exception as e:
        errors += 1
        print(f"Failed to check {model_zoo_path} because of {e}.")

if errors > 0:
    print(f"All {len(config.models_info)} model(s) have been checked, but {errors} model(s) failed.")
    sys.exit(1)
else:
    print(f"All {len(config.models_info)} model(s) have been checked.")
