import argparse
from mlagility_config import models_info
import os.path as osp
from os import listdir, rename
from pathlib import Path
import shutil
import subprocess
import sys
import ort_test_dir_utils


def get_immediate_subdirectories_count(dir_name):
    return len([name for name in listdir(dir_name)
            if osp.isdir(osp.join(dir_name, name))])


def find_model_hash_name(stdout):
    for line in stdout.decode().split("\n"):
        if "Build dir:" in line:
            # handle Windows path
            line = line.replace("\\", "/")
            # last part of the path is the model hash name
            return line.split("/")[-1]
    raise Exception(f"Cannot find Build dir in {stdout}.")


ZOO_OPSET_VERSION = "16"
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
    parser.add_argument("--drop", required=False, default=False, action="store_true",
                        help="Drop downloaded models after verification. (For space limitation in CIs)")

    args = parser.parse_args()
    errors = 0

    for model_info in models_info:
        _, model_name = model_info.split("/")
        model_name = model_name.replace(".py", "")
        model_zoo_dir = model_name
        print(f"----------------Checking {model_zoo_dir}----------------")
        final_model_dir = osp.join(mlagility_models_dir, model_zoo_dir)
        final_model_name = f"{model_zoo_dir}-{ZOO_OPSET_VERSION}.onnx"
        final_model_path = osp.join(final_model_dir, final_model_name)
        if osp.exists(final_model_path) and args.skip:
            print(f"Skip checking {model_zoo_dir} because {final_model_path} already exists.")
            continue
        try:
            cmd = subprocess.run(["benchit", osp.join(mlagility_root, model_info), "--cache-dir", cache_converted_dir,
                            "--onnx-opset", ZOO_OPSET_VERSION, "--export-only"],
                            cwd=cwd_path, stdout=subprocess.PIPE,
                            stderr=sys.stderr, check=True)
            model_hash_name = find_model_hash_name(cmd.stdout)
            mlagility_created_onnx = osp.join(cache_converted_dir, model_hash_name, "onnx", model_hash_name + base_name)
            if args.create:
                ort_test_dir_utils.create_test_dir(mlagility_created_onnx, "./", final_model_dir)
                rename(osp.join(final_model_dir, model_hash_name + base_name), final_model_path)
                print(f"Successfully created {model_zoo_dir} by mlagility and ORT.")
            else:
                shutil.copy(mlagility_created_onnx, final_model_path)
                """
                subprocess.run(["git", "diff", "--exit-code", "--", final_model_path],
                                cwd=cwd_path, stdout=sys.stdout,
                                stderr=sys.stderr, check=True)
                """
                ort_test_dir_utils.run_test_dir(final_model_dir)
                print(f"Successfully checked {model_zoo_dir} by mlagility.")
        except Exception as e:
            errors += 1
            print(f"Failed to check {model_zoo_dir} because of {e}.")
        if args.drop:
            subprocess.run(["benchit", "cache", "delete", "--all", "--cache-dir", cache_converted_dir], 
                        cwd=cwd_path, stdout=sys.stdout, stderr=sys.stderr, check=True)
            subprocess.run(["benchit", "cache", "clean", "--all", "--cache-dir", cache_converted_dir], 
                        cwd=cwd_path, stdout=sys.stdout, stderr=sys.stderr, check=True)
            shutil.rmtree(final_model_dir, ignore_errors=True)
            shutil.rmtree(cache_converted_dir, ignore_errors=True)

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
