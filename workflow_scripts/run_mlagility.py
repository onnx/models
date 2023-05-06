import os.path as osp
from pathlib import Path
import shutil
import subprocess
import sys


vision_dir = "vision"
classification_dir = osp.join(vision_dir, "classification")
object_detection_dir = osp.join(vision_dir, "object_detection_segmentation")

models_info = [
    # (script_path, model_name, model_zoo_path)
    ("torch_hub/alexnet.py", "alexnet_torch_hub_2891f54c", osp .join(classification_dir, "alexnet/alexnet.onnx")),
    ("torchvision/fasterrcnn_resnet50_fpn_v2.py", "fasterrcnn_resnet50_fpn_v2_torchvision_ae446d48", osp.join(object_detection_dir, "faster-rcnn/fasterrcnn_resnet50_fpn_v2.onnx")),
]

cwd_path = Path.cwd()
mlagility_root = "mlagility/models"
mlagility_models_dir = "mlagility_models"
ZOO_OPSET_VERSION = "18"

errors = 0

for script_path, model_name, model_zoo_path in models_info:
    try:
        final_model_path = osp.join(mlagility_models_dir, model_zoo_path.replace(".onnx", "-" + ZOO_OPSET_VERSION + ".onnx"))
        subprocess.run(["benchit", osp.join(mlagility_root, script_path), "--cache-dir", mlagility_models_dir, "--onnx-opset", ZOO_OPSET_VERSION],
                        cwd=cwd_path, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
        shutil.copy(osp.join(mlagility_models_dir, model_name), final_model_path)
        subprocess.run(["git", "diff", "--exit-code", "--", final_model_path],
                        cwd=cwd_path, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
        print(f"Successfully checked {model_zoo_path}.")
    except Exception as e:
        errors += 1
        print(f"Failed to check {model_zoo_path} because of {e}.")

if errors > 0:
    print(f"All {len(models_info)} models have been checked, but {errors} model(s) failed.")
    sys.exit(1)
else:
    print(f"All {len(models_info)} models have been checked.")
