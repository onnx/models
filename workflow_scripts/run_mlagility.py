import os.path as osp
from pathlib import Path
import subprocess
import sys


models_info = [
    # (script_path, model_name)
    ("torch_hub/alexnet.py", "alexnet_torch_hub_2891f54c"),
    ("torchvision/fasterrcnn_resnet50_fpn_v2.py", "fasterrcnn_resnet50_fpn_v2_torchvision_ae446d48"),
]

cwd_path = Path.cwd()
mlagility_root = "mlagility/models"
mlagility_models_dir = "mlagility_models"
ZOO_OPSET_VERSION = "18"

errors = 0

for model_info in models_info:
    try:
        subprocess.run(["benchit", osp.join(mlagility_root, model_info[0]), "--cache-dir", mlagility_models_dir, "--onnx-opset", ZOO_OPSET_VERSION],
                        cwd=cwd_path, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
        subprocess.run(["git", "diff", "--exit-code", "--", osp.join(mlagility_models_dir, model_info[1])],
                        cwd=cwd_path, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
        print(f"Successfully checked {model_info[1]}.")
    except:
        errors += 1
        print(f"Failed to check {model_info[1]}.")

if errors > 0:
    print(f"All {len(models_info)} models have been checked, but {errors} model(s) failed")
    sys.exit(1)
else:
    print(f"All {len(models_info)} models have been checked. ")
