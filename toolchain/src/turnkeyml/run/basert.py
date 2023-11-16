import os
import pathlib
import json
import shutil
import subprocess
from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod
import torch
import numpy as np
from turnkeyml.common.performance import MeasuredPerformance, Device
import turnkeyml.common.build as build
import turnkeyml.common.exceptions as exp
from turnkeyml.common.filesystem import Stats, rebase_cache_dir


def _check_docker_install():
    # Check if docker and python are installed on the local machine
    docker_location = shutil.which("docker")
    if not docker_location:
        raise ValueError("Docker installation not found. Please install Docker>=20.10")


def _check_docker_running():
    try:
        # On both Windows and Linux, 'docker info' will return an error if the
        #  Docker daemon is not running
        subprocess.run(
            ["docker", "info"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        raise ValueError(
            "Docker engine is not running. Please start the Docker engine."
        )


class BaseRT(ABC):
    def __init__(
        self,
        cache_dir: str,
        build_name: str,
        stats: Stats,
        device_type: Union[str, Device],
        runtime: str,
        runtimes_supported: List[str],
        runtime_version: str,
        base_path: str,
        iterations: int = 100,
        model: Optional[torch.nn.Module] = None,
        inputs: Optional[Dict[str, Any]] = None,
        requires_docker: bool = False,
        tensor_type=np.array,
        execute_function: Optional[callable] = None,
    ):
        self.tensor_type = tensor_type
        self.cache_dir = cache_dir
        self.build_name = build_name
        self.stats = stats
        if isinstance(device_type, Device):
            self.device_type = device_type
        else:
            self.device_type = Device(device_type)
        self.runtime = runtime
        self.iterations = iterations
        self.requires_docker = requires_docker
        self.conda_script = os.path.join(base_path, "within_conda.py")
        self.conda_requirements = os.path.join(base_path, "requirements.txt")
        self.docker_output_dir = "/app"
        self.runtime_version = runtime_version
        self.model = model
        self.inputs = inputs
        self.onnx_filename = "model.onnx"
        self.onnx_dirname = "onnxmodel"
        self.outputs_filename = "outputs.json"
        self.runtimes_supported = runtimes_supported
        self.execute_function = execute_function

        # Validate runtime is supported
        if runtime not in runtimes_supported:
            raise ValueError(
                f"'runtime' argument {runtime} passed to TensorRT, which only "
                f"supports runtimes: {runtimes_supported}"
            )

        self._setup()

    def posix_path_format(self, path) -> str:
        """
        Reformat a path into posix format. Necessary for when a path is
        created with os.path.join() on Windows that will be used in a posix
        environment such as a Docker container.
        """
        return pathlib.Path(path).as_posix()

    @property
    def local_output_dir(self):
        return os.path.join(
            build.output_dir(self.cache_dir, self.build_name),
            f"{str(self.device_type).replace('::','_')}_benchmark",
        )

    @property
    def local_onnx_dir(self):
        return os.path.join(self.local_output_dir, self.onnx_dirname)

    @property
    def docker_onnx_dir(self):
        return self.posix_path_format(
            os.path.join(self.docker_output_dir, self.onnx_dirname)
        )

    @property
    def local_onnx_file(self):
        return os.path.join(self.local_onnx_dir, self.onnx_filename)

    @property
    def docker_onnx_file(self):
        return self.posix_path_format(
            os.path.join(self.docker_onnx_dir, self.onnx_filename)
        )

    @property
    def local_outputs_file(self):
        return os.path.join(self.local_output_dir, self.outputs_filename)

    @property
    def docker_outputs_file(self):
        return self.posix_path_format(
            os.path.join(self.docker_output_dir, self.outputs_filename)
        )

    def _transfer_files(self, files_to_transfer: List[str]):
        """
        output_dir: destination for files
        files_to_transfer: absolute paths to files
        """

        os.makedirs(self.local_output_dir, exist_ok=True)
        for file in files_to_transfer:
            shutil.copy(
                file, os.path.join(self.local_output_dir, os.path.basename(file))
            )

    @abstractmethod
    def _setup(self) -> None:
        """
        Any code that should be called prior to benchmarking as a one-time setup.
        Called automatically at the end of  `BaseRT.__init__()`.
        """

    def _execute(self, output_dir, onnx_file, outputs_file) -> None:
        """
        Execute the benchmark in hardware using the selected runtime. Must produce output
        artifacts that can be parsed by the other methods of this class.
        Overloading this method is optional; the default implementation of benchmark()
        calls this method, however you are free to overload benchmark() to behave differently.
        """
        raise NotImplementedError

    def benchmark(self) -> MeasuredPerformance:
        """
        Transfer input artifacts, execute model on hardware, analyze output artifacts,
        and return the performance.
        """

        # Remove previous benchmarking artifacts
        if os.path.exists(self.local_outputs_file):
            os.remove(self.local_outputs_file)

        # Transfer input artifacts
        state = build.load_state(self.cache_dir, self.build_name)

        # Just in case the model file was generated on a different machine:
        # strip the state's cache dir, then prepend the current cache dir
        model_file = rebase_cache_dir(
            state.results[0], state.config.build_name, self.cache_dir
        )

        if not os.path.exists(model_file):
            msg = "Model file not found"
            raise exp.ModelRuntimeError(msg)

        os.makedirs(self.local_output_dir, exist_ok=True)
        os.makedirs(self.local_onnx_dir, exist_ok=True)
        shutil.copy(model_file, self.local_onnx_file)

        # Execute benchmarking in hardware
        if self.requires_docker:
            _check_docker_install()
            onnx_file = self.docker_onnx_file
            _check_docker_running()
        else:
            onnx_file = self.local_onnx_file

        self._execute(
            output_dir=self.local_output_dir,
            onnx_file=onnx_file,
            outputs_file=self.local_outputs_file,
        )

        if not os.path.isfile(self.local_outputs_file):
            raise exp.BenchmarkException(
                "No benchmarking outputs file found after benchmarking run. "
                "Sorry we don't have more information."
            )

        # Call property methods to analyze the output artifacts for performance stats
        # and return them
        return MeasuredPerformance(
            mean_latency=self.mean_latency,
            throughput=self.throughput,
            device=self.device_name,
            device_type=self.device_type,
            runtime=self.runtime,
            runtime_version=self.runtime_version,
            build_name=self.build_name,
        )

    def _get_stat(self, stat):
        if os.path.exists(self.local_outputs_file):
            with open(self.local_outputs_file, encoding="utf-8") as f:
                performance = json.load(f)
            return performance[stat]
        else:
            raise exp.BenchmarkException(
                "No benchmarking outputs file found after benchmarking run."
                "Sorry we don't have more information."
            )

    @property
    @abstractmethod
    def mean_latency(self) -> float:
        """
        Returns the mean latency, in ms, for the benchmarking run.
        """

    @property
    @abstractmethod
    def throughput(self) -> float:
        """
        Returns the throughput, in IPS, for the benchmarking run.
        """

    @property
    @abstractmethod
    def device_name(self) -> str:
        """
        Returns the full device name for the device used in benchmarking.
        For example, a benchmark on a `x86` device might have a device name like
        `AMD Ryzen 7 PRO 6850U with Radeon Graphics`.
        """
