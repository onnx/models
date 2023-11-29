import subprocess
import os
import threading
import json
import numpy as np
from turnkeyml.run.tensorrt.execute import TRT_VERSION
from turnkeyml.run.basert import BaseRT
import turnkeyml.common.exceptions as exp
from turnkeyml.common.filesystem import Stats
from turnkeyml.run.tensorrt.execute import (
    measure_power,
    run,
    average_power_and_utilization,
)

def _get_nvidia_driver_version():
    try:
        output = subprocess.check_output(["nvidia-smi"], text=True)

        # Search for the driver version in the output
        for line in output.split("\n"):
            if "Driver Version" in line:
                # Extract and return the driver version
                return line.split(":")[1].strip().split()[0]

    except Exception as e: # pylint: disable=broad-except
        return str(e)

    return "Driver not found"
class TensorRT(BaseRT):
    def __init__(
        self,
        cache_dir: str,
        build_name: str,
        stats: Stats,
        iterations: int,
        device_type: str,
        runtime: str = "trt",
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
            runtimes_supported=["trt"],
            runtime_version=TRT_VERSION,
            base_path=os.path.dirname(__file__),
            model=model,
            inputs=inputs,
            requires_docker=True,
        )

    def _setup(self) -> None:
        # Check if at least one NVIDIA GPU is available locally
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            check=False,
        )

        if "NVIDIA" not in result.stdout or result.returncode == 1:
            msg = "No NVIDIA GPUs available on the local machine"
            raise exp.BenchmarkException(msg)

    def _execute(
        self,
        output_dir: str,
        onnx_file: str,
        outputs_file: str,
    ):
        power_readings = []
        average_power = 0

        # Start power measurement in a separate thread
        start_event = threading.Event()
        stop_event = threading.Event()
        power_thread = threading.Thread(
            target=measure_power, args=(start_event, stop_event, power_readings, 0.01)
        )

        # Add the GPU driver version to the stats file before execution
        gpu_driver_version = _get_nvidia_driver_version()
        self.stats.add_build_stat("gpu_driver_version", gpu_driver_version)
        power_thread.start()

        run(
            output_dir=output_dir,
            onnx_file=onnx_file,
            outputs_file=outputs_file,
            errors_file=os.path.join(output_dir, "nvidia_error_log.txt"),
            iterations=self.iterations,
            start_event=start_event,
            stop_event=stop_event,
        )

        # Wait for power measurement to finish
        power_thread.join()

        # Calculate the average power consumption, average utilization, and peak power consumption
        average_power, average_utilization = average_power_and_utilization(
            power_readings
        )
        peak_power = (
            max([reading[1] for reading in power_readings]) if power_readings else None
        )

        # Load existing GPU performance data
        with open(outputs_file, "r", encoding="utf-8") as out_file:
            gpu_performance = json.load(out_file)

        # Add average power consumption, average utilization,
        # and peak power consumption to the dictionary
        gpu_performance["Average power consumption (W)"] = (
            round(average_power, 2) if average_power is not None else None
        )
        gpu_performance["Peak power consumption (W)"] = (
            round(peak_power, 2) if peak_power is not None else None
        )
        gpu_performance["Average GPU utilization (%)"] = (
            round(average_utilization, 2) if average_utilization is not None else None
        )

        # Save the updated GPU performance data
        with open(outputs_file, "w", encoding="utf-8") as out_file:
            json.dump(gpu_performance, out_file, ensure_ascii=False, indent=4)

    @property
    def mean_latency(self):
        return float(self._get_stat("Total Latency")["mean "].split(" ")[1])

    @property
    def throughput(self):
        return float(self._get_stat("Throughput").split(" ")[0])

    @property
    def device_name(self):
        return self._get_stat("Selected Device")
