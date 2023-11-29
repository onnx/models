import os
from typing import Dict, Any
from statistics import mean
import time
from packaging import version
import torch
import numpy as np
from turnkeyml.run.basert import BaseRT
from turnkeyml.common.performance import MeasuredPerformance
from turnkeyml.run.onnxrt.execute import get_cpu_specs
import turnkeyml.build.ignition as ignition
import turnkeyml.common.build as build
import turnkeyml.common.exceptions as exp
from turnkeyml.common.filesystem import Stats


class TorchRT(BaseRT):
    def __init__(
        self,
        cache_dir: str,
        build_name: str,
        stats: Stats,
        device_type: str,
        runtime: str,
        iterations: int,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        tensor_type=np.array,
    ):
        self.throughput_ips = None
        self.mean_latency_ms = None

        super().__init__(
            cache_dir=cache_dir,
            build_name=build_name,
            stats=stats,
            device_type=device_type,
            runtime=runtime,
            iterations=iterations,
            runtimes_supported=["torch-eager", "torch-compiled"],
            runtime_version=str(torch.__version__),
            base_path=os.path.dirname(__file__),
            tensor_type=tensor_type,
            model=model,
            inputs=inputs,
        )

    def _setup(self) -> None:
        # Ensure we have the correct model type
        model_type = ignition.identify_model_type(self.model)
        if model_type != build.ModelType.PYTORCH:
            raise exp.IntakeError(
                f"Only Pytorch models are valid when runtime is {self.runtime}"
            )

        # Compile the model
        if self.runtime == "torch-compiled":
            # First ensure we have the required version of Pytorch
            clean_torch_version = self.runtime_version.split("+")[0]
            if version.parse(clean_torch_version) < version.parse("2.0.0"):
                exp.BenchmarkException(
                    (
                        f"{self.runtime} can only be used with Pytorch 2.0.0 or above. "
                        f"However, version {self.runtime_version} was found."
                    )
                )

            self.model = torch.compile(self.model)

    def benchmark(self) -> MeasuredPerformance:
        per_iteration_latency = [0] * self.iterations
        for idx in range(self.iterations):
            start_time = time.perf_counter()
            self.model(**self.inputs)
            end_time = time.perf_counter()
            per_iteration_latency[idx] = end_time - start_time

        # Calculate perf from per_iteration_latency
        self.mean_latency_ms = mean(per_iteration_latency) * 1000
        self.throughput_ips = float(
            1 / (np.sum(per_iteration_latency) / self.iterations)
        )

        return MeasuredPerformance(
            mean_latency=self.mean_latency,
            throughput=self.throughput,
            device=self.device_name,
            device_type=self.device_type,
            runtime=self.runtime,
            runtime_version=self.runtime_version,
            build_name=self.build_name,
        )

    @property
    def mean_latency(self) -> float:
        if self.mean_latency_ms is not None:
            return self.mean_latency_ms
        else:
            raise exp.BenchmarkException(
                "Queried mean latency before self.benchmark() was called"
            )

    @property
    def throughput(self) -> float:
        if self.throughput_ips is not None:
            return self.throughput_ips
        else:
            raise exp.BenchmarkException(
                "Queried throughput before self.benchmark() was called"
            )

    @property
    def device_name(self) -> str:
        return get_cpu_specs()["CPU Name"]
