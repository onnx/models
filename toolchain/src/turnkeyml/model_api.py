from typing import Any, Dict, Optional, Union, List
from turnkeyml.build_api import build_model
from turnkeyml.build.stage import Sequence
import turnkeyml.common.printing as printing
import turnkeyml.common.filesystem as filesystem
from turnkeyml.common.performance import MeasuredPerformance
from turnkeyml.run.devices import (
    SUPPORTED_DEVICES,
    SUPPORTED_RUNTIMES,
    DEVICE_RUNTIME_MAP,
    apply_default_runtime,
)
import turnkeyml.build.sequences as sequences
import turnkeyml.common.exceptions as exp

TURNKEY_DEFAULT_REBUILD_POLICY = "if_needed"


def benchmark_model(
    model: Any,
    inputs: Dict[str, Any],
    build_name: str,
    iterations: int = 100,
    stats_id: str = "build",
    cache_dir: str = filesystem.DEFAULT_CACHE_DIR,
    device: str = "x86",
    runtime: Optional[str] = None,
    build_only: bool = False,
    lean_cache: bool = False,
    rebuild: str = TURNKEY_DEFAULT_REBUILD_POLICY,
    onnx_opset: Optional[int] = None,
    sequence: Sequence = None,
    rt_args: Optional[Dict[str, Union[str, List[str]]]] = None,
) -> MeasuredPerformance:
    """
    Benchmark a model against some inputs on target hardware
    """

    selected_runtime = apply_default_runtime(device, runtime)

    # Build and benchmark the model
    try:
        # Validate device and runtime selections
        if device not in SUPPORTED_DEVICES:
            raise exp.ArgError(
                f"Device argument '{device}' is not one of the available "
                f"supported devices {SUPPORTED_DEVICES}\n"
                f"You may need to check the spelling of '{device}', install a "
                "plugin, or update the turnkeyml package."
            )
        else:
            if selected_runtime not in DEVICE_RUNTIME_MAP[device]:
                raise exp.ArgError(
                    f"Runtime argument '{selected_runtime}' is not one of the available "
                    f"runtimes supported for device '{device}': {DEVICE_RUNTIME_MAP[device]}\n"
                    f"You may need to check the spelling of '{selected_runtime}', install a "
                    "plugin, or update the turnkeyml package."
                )

        # Get the plugin module for the selected runtime
        runtime_info = SUPPORTED_RUNTIMES[selected_runtime]

        # Perform a build, if necessary
        if runtime_info["build_required"]:
            # Get the build sequence that will be used for the model
            if sequence is None:
                # Automatically choose a Sequence based on what the runtime expects
                sequence_selected = runtime_info["default_sequence"]
            else:
                # User-specified Sequence
                if isinstance(sequence, str):
                    # Sequence is defined by a plugin
                    if sequence in sequences.SUPPORTED_SEQUENCES.keys():
                        sequence_selected = sequences.SUPPORTED_SEQUENCES[sequence]
                    else:
                        raise ValueError(
                            f"Sequence argument {sequence} is not one of the "
                            "available sequences installed: "
                            f"{sequences.SUPPORTED_SEQUENCES.keys()} \n"
                            f"You may need to check the spelling of `{sequence}`, "
                            "install a plugin, or update the turnkeyml package."
                        )

                elif isinstance(sequence, Sequence):
                    # Sequence is a user-defined instance of Sequence
                    sequence_selected = sequence

            build_model(
                model=model,
                inputs=inputs,
                stats_id=stats_id,
                build_name=build_name,
                cache_dir=cache_dir,
                rebuild=rebuild,
                sequence=sequence_selected,
                onnx_opset=onnx_opset,
                device=device,
            )

        # Perform benchmarking, if requested
        if not build_only:
            if rt_args is None:
                rt_args_to_use = {}
            else:
                rt_args_to_use = rt_args

            printing.log_info(f"Benchmarking on {device}...")
            stats = filesystem.Stats(cache_dir, build_name, stats_id)
            model_handle = runtime_info["RuntimeClass"](
                cache_dir=cache_dir,
                build_name=build_name,
                stats=stats,
                iterations=iterations,
                model=model,
                inputs=inputs,
                device_type=device,
                runtime=selected_runtime,
                **rt_args_to_use,
            )
            perf = model_handle.benchmark()

    finally:
        # Make sure the build and cache dirs exist and have the proper marker files
        # NOTE: We would do this at the top of the file, however
        # there are conditions where build_model() will wipe the build dir,
        # which would eliminate our marker file
        filesystem.make_build_dir(cache_dir, build_name)

        # Clean cache if needed
        if lean_cache:
            printing.log_info("Removing build artifacts...")
            filesystem.clean_output_dir(cache_dir, build_name)

    if not build_only:
        perf.print()
        return perf
    else:
        return None
