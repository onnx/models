import os
import numpy as np
from turnkeyml.run.basert import BaseRT
from turnkeyml.common.performance import MeasuredPerformance
import turnkeyml.common.exceptions as exp
from turnkeyml.common.filesystem import Stats

example_rt_name = "example-rt"


class ExampleRT(BaseRT):
    def __init__(
        self,
        cache_dir: str,
        build_name: str,
        stats: Stats,
        iterations: int,
        device_type: str,
        runtime: str = example_rt_name,
        tensor_type=np.array,
        model=None,
        inputs=None,
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
            runtimes_supported=[example_rt_name],
            runtime_version="0.0.0",
            base_path=os.path.dirname(__file__),
            tensor_type=tensor_type,
            model=model,
            inputs=inputs,
        )

    def _setup(self):
        # The BaseRT abstract base class requires us to overload this function,
        # however our simple example runtime does not require any additional
        # setup steps.
        pass

    def benchmark(self) -> MeasuredPerformance:
        self.throughput_ips = self.iterations
        self.mean_latency_ms = 1 / self.iterations

        # Assign values to the stats that will be printed
        # out by the CLI when status is reported
        self.stats.add_build_stat("magic_perf_points", 42)
        self.stats.add_build_stat("super_runtime_points", 100)

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
        return "the x86 cpu of your dreams"
