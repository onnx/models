import platform
import os
import numpy as np
from turnkeyml.run.basert import BaseRT
import turnkeyml.common.exceptions as exp
from turnkeyml.run.onnxrt.execute import ORT_VERSION
from turnkeyml.common.filesystem import Stats
from turnkeyml.run.onnxrt.execute import create_conda_env, execute_benchmark
import turnkeyml.run.plugin_helpers as plugin_helpers


class OnnxRT(BaseRT):
    def __init__(
        self,
        cache_dir: str,
        build_name: str,
        stats: Stats,
        iterations: int,
        device_type: str,
        runtime: str = "ort",
        tensor_type=np.array,
        model=None,
        inputs=None,
    ):
        super().__init__(
            cache_dir=cache_dir,
            build_name=build_name,
            stats=stats,
            tensor_type=tensor_type,
            device_type=device_type,
            iterations=iterations,
            runtime=runtime,
            runtimes_supported=["ort"],
            runtime_version=ORT_VERSION,
            base_path=os.path.dirname(__file__),
            model=model,
            inputs=inputs,
            requires_docker=False,
        )

    def _setup(self):
        # Check if x86_64 (aka AMD64) CPU is available locally
        machine = platform.uname().machine
        if machine != "x86_64" and machine != "AMD64":
            msg = "Only x86_64 and AMD64 CPUs are supported, got {machine}"
            raise exp.ModelRuntimeError(msg)

        self._transfer_files([self.conda_script])

    def _execute(
        self,
        output_dir: str,
        onnx_file: str,
        outputs_file: str,
    ):
        conda_env_name = "turnkey-onnxruntime-mlas-ep"

        try:
            # Create and setup the conda env
            create_conda_env(conda_env_name)
        except PermissionError as pe:
            os_type = platform.system()
            if os_type == "Windows":
                raise plugin_helpers.CondaError(
                    f"Conda environment setup encountered a permission issue: {pe}. "
                    "Ensure you have write permissions for the Conda installation directory. "
                    "If Conda is installed for 'All Users' on a Windows machine, it defaults "
                    "to 'C:\\ProgramData', where you may not have the necessary permissions."
                    "To resolve this, consider reinstalling Conda for 'Just Me' instead."
                )
            else:
                raise plugin_helpers.CondaError(
                    f"Conda env setup failed due to permission error: {pe}"
                )
        except Exception as e:
            raise plugin_helpers.CondaError(
                f"Conda env setup failed with exception: {e}"
            )

        # Execute the benchmark script in the conda environment
        execute_benchmark(
            onnx_file=onnx_file,
            outputs_file=outputs_file,
            output_dir=output_dir,
            conda_env_name=conda_env_name,
            iterations=self.iterations,
        )

    @property
    def mean_latency(self):
        return float(self._get_stat("Mean Latency(ms)"))

    @property
    def throughput(self):
        return float(self._get_stat("Throughput"))

    @property
    def device_name(self):
        return self._get_stat("CPU Name")
