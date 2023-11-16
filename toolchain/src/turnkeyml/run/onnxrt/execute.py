"""
The following script is used to get the latency and outputs of a given run on the x86 CPUs.
"""
# pylint: disable = no-name-in-module
# pylint: disable = import-error
import os
import subprocess
import json
from statistics import mean
import platform
import turnkeyml.common.build as build
import turnkeyml.run.plugin_helpers as plugin_helpers

ORT_VERSION = "1.15.1"

BATCHSIZE = 1


def create_conda_env(conda_env_name: str):
    """Create a Conda environment with the given name and install requirements."""
    conda_path = os.getenv("CONDA_EXE")
    if conda_path is None:
        raise EnvironmentError(
            "CONDA_EXE environment variable not set."
            "Make sure Conda is properly installed."
        )

    # Normalize the path for Windows
    if platform.system() == "Windows":
        conda_path = os.path.normpath(conda_path)

    env_path = os.path.join(
        os.path.dirname(os.path.dirname(conda_path)), "envs", conda_env_name
    )

    # Only create the environment if it does not already exist
    if not os.path.exists(env_path):
        plugin_helpers.run_subprocess(
            [
                conda_path,
                "create",
                "--name",
                conda_env_name,
                "python=3.8",
                "-y",
            ]
        )

    # Using conda run to execute pip install within the environment
    setup_cmd = [
        conda_path,
        "run",
        "--name",
        conda_env_name,
        "pip",
        "install",
        f"onnxruntime=={ORT_VERSION}",
    ]
    plugin_helpers.run_subprocess(setup_cmd)


def execute_benchmark(
    onnx_file: str,
    outputs_file: str,
    output_dir: str,
    conda_env_name: str,
    iterations: int,
):
    """Execute the benchmark script and retrieve the output."""

    python_in_env = plugin_helpers.get_python_path(conda_env_name)
    iterations_file = os.path.join(output_dir, "per_iteration_latency.json")
    benchmarking_log_file = os.path.join(output_dir, "ort_benchmarking_log.txt")

    cmd = [
        python_in_env,
        os.path.join(output_dir, "within_conda.py"),
        "--onnx-file",
        onnx_file,
        "--iterations",
        str(iterations),
        "--iterations-file",
        iterations_file,
    ]

    # Execute command and log stdout/stderr
    build.logged_subprocess(
        cmd=cmd,
        cwd=os.path.dirname(output_dir),
        log_to_std_streams=False,
        log_to_file=True,
        log_file_path=benchmarking_log_file,
    )

    # Parse per-iteration performance results and save aggregated results to a json file
    if os.path.exists(iterations_file):
        with open(iterations_file, "r", encoding="utf-8") as f:
            per_iteration_latency = json.load(f)
    else:
        raise ValueError(
            f"Execution of command {cmd} failed, see {benchmarking_log_file}"
        )

    cpu_performance = get_cpu_specs()
    cpu_performance["OnnxRuntime Version"] = str(ORT_VERSION)
    cpu_performance["Mean Latency(ms)"] = str(mean(per_iteration_latency) * 1000)
    cpu_performance["Throughput"] = str(BATCHSIZE / mean(per_iteration_latency))
    cpu_performance["Min Latency(ms)"] = str(min(per_iteration_latency) * 1000)
    cpu_performance["Max Latency(ms)"] = str(max(per_iteration_latency) * 1000)

    with open(outputs_file, "w", encoding="utf-8") as out_file:
        json.dump(cpu_performance, out_file, ensure_ascii=False, indent=4)


def get_cpu_specs() -> dict:
    # Define a common field mapping for both Windows and Linux
    field_mapping = {
        "Architecture": "CPU Architecture",
        "Manufacturer": "CPU Vendor",
        "MaxClockSpeed": "CPU Max Frequency (MHz)",
        "Name": "CPU Name",
        "NumberOfCores": "CPU Core Count",
        "Model name": "CPU Name",  # Additional mapping for Linux
        "CPU MHz": "CPU Max Frequency (MHz)",  # Additional mapping for Linux
        "CPU(s)": "CPU Core Count",  # Additional mapping for Linux
        "Vendor ID": "CPU Vendor",
    }

    # Check the operating system and define the command accordingly
    if platform.system() == "Windows":
        cpu_info_command = (
            "wmic CPU get Architecture,Manufacturer,MaxClockSpeed,"
            "Name,NumberOfCores /format:list"
        )

        cpu_info = subprocess.Popen(
            cpu_info_command, stdout=subprocess.PIPE, shell=True
        )
        separator = "="
    else:
        cpu_info_command = "lscpu"
        cpu_info = subprocess.Popen(cpu_info_command.split(), stdout=subprocess.PIPE)
        separator = ":"

    cpu_info_output, _ = cpu_info.communicate()
    if not cpu_info_output:
        raise EnvironmentError(
            f"Could not get CPU info using '{cpu_info_command.split()[0]}'. "
            "Please make sure this tool is correctly installed on your system before continuing."
        )

    decoded_info = (
        cpu_info_output.decode()
        .strip()
        .split("\r\n" if platform.system() == "Windows" else "\n")
    )

    # Initialize an empty dictionary to hold the CPU specifications
    cpu_spec = {}
    for line in decoded_info:
        key, value = line.split(separator, 1)
        # Get the corresponding key from the field mapping
        key = field_mapping.get(key.strip())
        if key:
            # Add the key and value to the CPU specifications dictionary
            cpu_spec[key] = value.strip()

    return cpu_spec
