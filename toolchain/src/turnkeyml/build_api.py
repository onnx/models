import os
from typing import Optional, List, Dict, Any
from collections.abc import Collection
import turnkeyml.build.ignition as ignition
import turnkeyml.build.stage as stage
import turnkeyml.common.printing as printing
import turnkeyml.common.build as build
import turnkeyml.common.filesystem as filesystem


def build_model(
    model: build.UnionValidModelInstanceTypes = None,
    inputs: Optional[Dict[str, Any]] = None,
    build_name: Optional[str] = None,
    stats_id: Optional[str] = "build",
    cache_dir: str = filesystem.DEFAULT_CACHE_DIR,
    monitor: Optional[bool] = None,
    rebuild: Optional[str] = None,
    sequence: Optional[List[stage.Stage]] = None,
    quantization_samples: Collection = None,
    onnx_opset: Optional[int] = None,
    device: Optional[str] = None,
) -> build.State:
    """Use build a model instance into an optimized ONNX file.

    Args:
        model: Model to be mapped to an optimized ONNX file, which can be a PyTorch
            model instance, Keras model instance, Hummingbird model instance,
            or a path to an ONNX file.
        inputs: Example inputs to the user's model. The ONNX file will be
            built to handle inputs with the same static shape only.
        build_name: Unique name for the model that will be
            used to store the ONNX file and build state on disk. Defaults to the
            name of the file that calls build_model().
        stats_id: Unique name for build statistics that should persist across multiple
            builds of the same model.
        cache_dir: Directory to use as the cache for this build. Output files
            from this build will be stored at cache_dir/build_name/
            Defaults to the current working directory, but we recommend setting it to
            an absolute path of your choosing.
        monitor: Display a monitor on the command line that
            tracks the progress of this function as it builds the ONNX file.
        rebuild: determines whether to rebuild or load a cached build. Options:
            - "if_needed" (default): overwrite invalid cached builds with a rebuild
            - "always": overwrite valid cached builds with a rebuild
            - "never": load cached builds without checking validity, with no guarantee
                of functionality or correctness
            - None: Falls back to default
        sequence: Override the default sequence of build stages. Power
            users only.
        quantization_samples: If set, performs post-training quantization
            on the ONNX model using the provided samplesIf the previous build used samples
            that are different to the samples used in current build, the "rebuild"
            argument needs to be manually set to "always" in the current build
            in order to create a new ONNX file.
        onnx_opset: ONNX opset to use during ONNX export.
        device: Specific device target to take into account during the build sequence.
            Use the format "device_family", "device_family::part", or
            "device_family::part::configuration" to refer to a family of devices,
            part within a family, or configuration of a part model, respectively.

        More information is available in the Tools User Guide:
            https://github.com/aigdat/onnxmodelzoo/blob/main/toolchain/docs/tools_user_guide.md
    """

    # Allow monitor to be globally disabled by an environment variable
    if monitor is None:
        if os.environ.get("TURNKEY_BUILD_MONITOR") == "False":
            monitor_setting = False
        else:
            monitor_setting = True
    else:
        monitor_setting = monitor

    # Support "~" in the cache_dir argument
    parsed_cache_dir = os.path.expanduser(cache_dir)

    # Validate and lock in the config (user arguments that
    # configure the build) that will be used by the rest of the toolchain
    config = ignition.lock_config(
        model=model,
        build_name=build_name,
        sequence=sequence,
        onnx_opset=onnx_opset,
        device=device,
    )

    # Analyze the user's model argument and lock in the model, inputs,
    # and sequence that will be used by the rest of the toolchain
    (
        model_locked,
        inputs_locked,
        sequence_locked,
        model_type,
    ) = ignition.model_intake(
        model,
        inputs,
        sequence,
        user_quantization_samples=quantization_samples,
    )

    # Get the state of the model from the cache if a valid build is available
    state = ignition.load_or_make_state(
        config=config,
        stats_id=stats_id,
        cache_dir=parsed_cache_dir,
        rebuild=rebuild or build.DEFAULT_REBUILD_POLICY,
        model_type=model_type,
        monitor=monitor_setting,
        model=model_locked,
        inputs=inputs_locked,
        quantization_samples=quantization_samples,
    )

    # Return a cached build if possible, otherwise prepare the model State for
    # a build
    if state.build_status == build.Status.SUCCESSFUL_BUILD:
        # Successful builds can be loaded from cache and returned with
        # no additional steps
        additional_msg = " (build_name auto-selected)" if config.auto_name else ""
        printing.log_success(
            f' Build "{config.build_name}"{additional_msg} found in cache. Loading it!',
        )

        return state

    state.quantization_samples = quantization_samples

    sequence_locked.show_monitor(config, state.monitor)
    state = sequence_locked.launch(state)

    if state.build_status == build.Status.SUCCESSFUL_BUILD:
        printing.log_success(
            f"\n    Saved to **{build.output_dir(state.cache_dir, config.build_name)}**"
        )

        return state

    else:
        printing.log_success(
            f"Build Sequence {sequence_locked.unique_name} completed successfully"
        )
        msg = """
        build_model() only returns a Model instance if the Sequence includes a Stage
        that sets state.build_status=turnkey.common.build.Status.SUCCESSFUL_BUILD.
        """
        printing.log_warning(msg)
