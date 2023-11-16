import os
import time
from timeit import default_timer as timer
import onnxruntime as ort
import numpy as np
from turnkeyml.run.basert import BaseRT
import turnkeyml.common.exceptions as exp
import turnkeyml.common.build as build
from turnkeyml.run.onnxrt.within_conda import dummy_inputs
from turnkeyml.common.performance import MeasuredPerformance, Device
from turnkeyml.common.filesystem import Stats


combined_rt_name = "example-combined-rt"

class CombinedExampleRT(BaseRT):
    def __init__(
        self,
        cache_dir: str,
        build_name: str,
        stats: Stats,
        iterations: int,
        device_type: str,
        runtime: str = combined_rt_name,
        tensor_type=np.array,
        model=None,
        inputs=None,
        delay_before_benchmarking: str = "0",
    ):
        # Custom runtime args always arive as strings, so we need to convert them
        # to the appropriate data type here
        self.delay_before_benchmarking = int(delay_before_benchmarking)

        super().__init__(
            cache_dir=cache_dir,
            build_name=build_name,
            stats=stats,
            tensor_type=tensor_type,
            device_type=device_type,
            runtime=runtime,
            iterations=iterations,
            runtimes_supported=[combined_rt_name],
            runtime_version="0.0.0",
            base_path=os.path.dirname(__file__),
            model=model,
            inputs=inputs,
        )

    def _setup(self):
        # The BaseRT abstract base class requires us to overload this function,
        # however our simple example runtime does not require any additional
        # setup steps.
        pass

    def benchmark(self):
        state = build.load_state(self.cache_dir, self.build_name)
        per_iteration_latency = []
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        onnx_session = ort.InferenceSession(state.results[0], sess_options)
        sess_input = onnx_session.get_inputs()
        input_feed = dummy_inputs(sess_input)
        output_name = onnx_session.get_outputs()[0].name

        # Using custom runtime argument
        print(f"Sleeping {self.delay_before_benchmarking}s before benchmarking")
        time.sleep(self.delay_before_benchmarking)

        for _ in range(self.iterations):
            start = timer()
            onnx_session.run([output_name], input_feed)
            end = timer()
            iteration_latency = end - start
            per_iteration_latency.append(iteration_latency)

        total_time = sum(per_iteration_latency)
        self.throughput_ips = total_time / self.iterations
        self.mean_latency_ms = 1 / self.throughput_ips

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
        return f"Device Family {self.device_type.family}, Device Part {self.device_type.part}, Device Configuration {self.device_type.config}"
