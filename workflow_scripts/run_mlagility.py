import config
import os.path as osp
from os import listdir
from pathlib import Path
import shutil
import subprocess
import sys

def get_immediate_subdirectories_count(dir):
    return len([name for name in listdir(dir)
            if osp.isdir(osp.join(dir, name))])


ZOO_OPSET_VERSION = "18"
base_name = f"-op{ZOO_OPSET_VERSION}-base.onnx"
cwd_path = Path.cwd()
mlagility_root = "mlagility/models"
mlagility_models_dir = "models/mlagility"
cache_converted_dir = ".cache"

errors = 0

for script_path, model_name, model_zoo_dir in config.models_info:
    try:
        print(f"----------------Checking {model_zoo_dir}----------------")
        final_model_path = osp.join(mlagility_models_dir, model_zoo_dir, f"{model_zoo_dir}-{ZOO_OPSET_VERSION}.onnx")
        subprocess.run(["benchit", osp.join(mlagility_root, script_path), "--cache-dir", cache_converted_dir,
                        "--onnx-opset", ZOO_OPSET_VERSION, "--export-only"],
                        cwd=cwd_path, stdout=sys.stdout,
                        stderr=sys.stderr)
        shutil.copy(osp.join(cache_converted_dir, model_name, "onnx", model_name + base_name), final_model_path)
        subprocess.run(["git", "diff", "--exit-code", "--", final_model_path],
                        cwd=cwd_path, stdout=sys.stdout,
                        stderr=sys.stderr)
        print(f"Successfully checked {model_zoo_dir}.")
    except Exception as e:
        errors += 1
        print(f"Failed to check {model_zoo_dir} because of {e}.")

if errors > 0:
    print(f"All {len(config.models_info)} model(s) have been checked, but {errors} model(s) failed.")
    sys.exit(1)
else:
    print(f"All {len(config.models_info)} model(s) have been checked.")

mlagility_subdir_count = get_immediate_subdirectories_count(mlagility_models_dir)
if mlagility_subdir_count != len(config.models_info):
    print(f"Expected {len(config.models_info)} model(s) in {mlagility_models_dir}, but got {mlagility_subdir_count} model(s) under models/mlagility.")
    sys.exit(1)

