from dataclasses import dataclass
from typing import Optional, Union, Dict
import turnkeyml.common.printing as printing
import turnkeyml.common.exceptions as exp


class Device:
    def __init__(self, selected_device: str, rt_supported_devices: Optional[Dict] = None):
        self.family: str
        self.part: Optional[str] = None
        self.config: Optional[str] = None

        # Unpack selected_device
        values = selected_device.split("::")
        if len(values) > 3:
            raise exp.ArgError(
                f"Recieved a device argument that has more than 3 members: {selected_device}. "
                "Please format device arguments as either `family` (1 member), "
                "`family::part` (2 members), or `family::part::config` (3 members)."
            )

        # Set family, part, and config straight away if rt_supported_devices is not provided
        if rt_supported_devices is None:
            if len(values)>0:
                self.family = values[0]
            if len(values)>1:
                self.part = values[1]
            if len(values)>2:
                self.config = values[2]
            return

        # If rt_supported_devices is provided we:
        # (1) Check whether the received family/part/config are in rt_supported_devices
        # (2) Set part/config to the default values (if needed)
        # Note: The default values are set according to the order in which elements were
        #       added to rt_supported_devices. The first element is set as the default.
        #       This feature relies on dictionaries being ordered (Python 3.6+).

        # Set family
        if values[0] in rt_supported_devices:
            self.family = values[0]
        else:
            raise exp.ArgError(
                f"Family {values[0]} is not supported by this device. "
                f"Supported families are: {rt_supported_devices.keys()}"
            )

        # Set part (assign to default if no needed)
        if len(values) > 1:
            if values[1] in rt_supported_devices[self.family]:
                self.part = values[1]
            elif len(rt_supported_devices[self.family]) == 0:
                raise exp.ArgError(
                    f"Device family {self.family} supports no parts."
                )
            else:
                error_msg = f"Part {values[1]} is not supported by this device family."
                if len(rt_supported_devices[self.family]) > 0:
                    error_msg += f" Supported parts are: {rt_supported_devices[self.family]}"
                raise exp.ArgError(error_msg)
        elif rt_supported_devices[self.family]:
            self.part = next(iter(rt_supported_devices[self.family]))

        # Set config (assign to default if no needed)
        if len(values) > 2:
            supported_configs = rt_supported_devices[self.family][self.part]
            if values[2] in supported_configs:
                self.config = values[2]
            else:
                error_msg = f"Config {values[2]} is not supported by this device family and part."
                if len(supported_configs) > 0:
                    error_msg += f"Supported configs are: {supported_configs}"
                raise exp.ArgError(error_msg)
        elif self.part is not None:
            if rt_supported_devices[self.family][self.part]:
                self.config = rt_supported_devices[self.family][self.part][0]

    def __str__(self) -> str:
        result = self.family

        if self.part:
            result = result + "::" + self.part

            if self.config:
                result = result + "::" + self.config

        return result


@dataclass
class MeasuredPerformance:
    throughput: float
    mean_latency: float
    device: str
    runtime: str
    runtime_version: str
    device_type: Union[str, Device]
    build_name: str
    throughput_units: str = "inferences per second (IPS)"
    latency_units: str = "milliseconds (ms)"

    def print(self):
        printing.log_info(
            f"\nPerformance of build {self.build_name} on {self.device} "
            f"({self.runtime} v{self.runtime_version}) is:"
        )
        print(f"\tMean Latency: {self.mean_latency:.3f} {self.latency_units}")
        print(f"\tThroughput: {self.throughput:.1f} {self.throughput_units}")
        print()

    def __post_init__(self):
        if isinstance(self.device_type, Device):
            self.device_type = str(self.device_type)
