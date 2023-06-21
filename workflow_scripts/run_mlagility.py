import argparse
from mlagility_config import models_info
import os.path as osp
from os import listdir
from pathlib import Path
import shutil
import subprocess
import sys


def get_immediate_subdirectories_count(dir_name):
    return len([name for name in listdir(dir_name)
            if osp.isdir(osp.join(dir_name, name))])


def find_model_hash_name(dir_name, cache_dir_prefix):
    for dir in listdir(dir_name):
        if dir.startswith(cache_dir_prefix):
            return dir
    raise Exception(f"Cannot find model hash name: {cache_dir_prefix} in cache directory.")


ZOO_OPSET_VERSION = "18"
base_name = f"-op{ZOO_OPSET_VERSION}-base.onnx"
cwd_path = Path.cwd()
mlagility_root = "mlagility/models"
mlagility_models_dir = "models/mlagility"
cache_converted_dir = ".cache"


def main():
    parser = argparse.ArgumentParser(description="Test settings")

    parser.add_argument("--create", required=False, default=False, action="store_true",
                        help="Create new models from mlagility if not exist.")
    parser.add_argument("--skip", required=False, default=False, action="store_true",
                        help="Skip checking models if already exist.")

    args = parser.parse_args()
    errors = 0

    for model_info in models_info:
        directory_name, model_name = model_info.split("/")
        model_name = model_name.replace(".py", "")
        model_zoo_dir = model_name
        try:
            print(f"----------------Checking {model_zoo_dir}----------------")
            final_model_path = osp.join(mlagility_models_dir, model_zoo_dir, f"{model_zoo_dir}-{ZOO_OPSET_VERSION}.onnx")
            if osp.exists(final_model_path) and args.skip:
                print(f"Skip checking {model_zoo_dir} because {final_model_path} already exists.")
                continue
            subprocess.run(["benchit", osp.join(mlagility_root, model_info), "--cache-dir", cache_converted_dir,
                            "--onnx-opset", ZOO_OPSET_VERSION, "--export-only"],
                            cwd=cwd_path, stdout=sys.stdout,
                            stderr=sys.stderr, check=True)
            model_hash_name = find_model_hash_name(".cache", model_name + "_" + directory_name + "_")
            shutil.copy(osp.join(cache_converted_dir, model_hash_name, "onnx", model_hash_name + base_name), final_model_path)
            if not args.create:
                print(f"Successfully created {model_zoo_dir} by mlagility.")
            else:
                subprocess.run(["git", "diff", "--exit-code", "--", final_model_path],
                                cwd=cwd_path, stdout=sys.stdout,
                                stderr=sys.stderr, check=True)
                print(f"Successfully checked {model_zoo_dir} by mlagility.")
        except Exception as e:
            errors += 1
            print(f"Failed to check {model_zoo_dir} because of {e}.")

    if errors > 0:
        print(f"All {len(models_info)} model(s) have been checked, but {errors} model(s) failed.")
        sys.exit(1)
    else:
        print(f"All {len(models_info)} model(s) have been checked.")

    mlagility_subdir_count = get_immediate_subdirectories_count(mlagility_models_dir)
    if mlagility_subdir_count != len(models_info):
        print(f"Expected {len(models_info)} model(s) in {mlagility_models_dir}, but got {mlagility_subdir_count} model(s) under models/mlagility."
            f"Please check if you have added new model(s) to models_info in mlagility_config.py.")
        sys.exit(1)


if __name__ == "__main__":
    main()
