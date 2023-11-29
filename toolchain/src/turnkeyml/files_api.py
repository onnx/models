import time
import os
import copy
import glob
import pathlib
from typing import Tuple, List, Dict, Optional, Union
import turnkeyml.common.printing as printing
import turnkeyml.common.exceptions as exceptions
import turnkeyml.build.stage as stage
import turnkeyml.cli.spawn as spawn
import turnkeyml.common.filesystem as filesystem
import turnkeyml.common.labels as labels_library
import turnkeyml.run.devices as devices
from turnkeyml.common.performance import Device
from turnkeyml.run.devices import SUPPORTED_RUNTIMES
from turnkeyml.analyze.script import (
    evaluate_script,
    TracerArgs,
    Action,
    explore_invocation,
    get_model_hash,
)
from turnkeyml.analyze.util import ModelInfo, UniqueInvocationInfo
import turnkeyml.common.build as build
import turnkeyml.build.onnx_helpers as onnx_helpers


def decode_input_arg(input: str) -> Tuple[str, List[str], str]:
    # Parse the targets out of the file name
    # Targets use the format:
    #   file_path.ext::target0,target1,...,targetN
    decoded_input = input.split("::")
    file_path = os.path.abspath(decoded_input[0])

    if len(decoded_input) == 2:
        targets = decoded_input[1].split(",")
        encoded_input = file_path + "::" + decoded_input[1]
    elif len(decoded_input) == 1:
        targets = []
        encoded_input = file_path
    else:
        raise ValueError(
            "Each file input to turnkey should have either 0 or 1 '::' in it."
            f"However, {file_path} was received."
        )

    return file_path, targets, encoded_input


def check_sequence_type(
    sequence: Union[str, stage.Sequence],
    use_slurm: bool,
    process_isolation: bool,
):
    """
    Check to make sure the user's sequence argument is valid.
    use_slurm or process_isolation: only work with names of installed sequences
    otherwise: sequence instances and sequence names are allowed
    """

    if sequence is not None:
        if use_slurm or process_isolation:
            # The spawned process will need to load a sequence file
            if not isinstance(sequence, str):
                raise ValueError(
                    "The 'sequence' arg must be a str (name of an installed sequence) "
                    "when use_slurm=True or process_isolation=True."
                )


def unpack_txt_inputs(input_files: List[str]) -> List[str]:
    """
    Replace txt inputs with models listed inside those files
    Note: This implementation allows for nested .txt files
    """
    txt_files_expanded = sum(
        [glob.glob(f) for f in input_files if f.endswith(".txt")], []
    )
    processed_files = []
    for input_string in txt_files_expanded:
        if not os.path.exists(input_string):
            raise exceptions.ArgError(
                f"{input_string} does not exist. Please verify the file."
            )

        with open(input_string, "r", encoding="utf-8") as file:
            lines = [line.strip() for line in file if line.strip() != ""]
            processed_files.extend(unpack_txt_inputs(lines))

    return processed_files + [f for f in input_files if not f.endswith(".txt")]


# pylint: disable=unused-argument
def benchmark_files(
    input_files: List[str],
    use_slurm: bool = False,
    process_isolation: bool = False,
    lean_cache: bool = False,
    cache_dir: str = filesystem.DEFAULT_CACHE_DIR,
    labels: List[str] = None,
    rebuild: Optional[str] = None,
    device: str = "x86",
    runtime: str = None,
    iterations: int = 100,
    analyze_only: bool = False,
    build_only: bool = False,
    script_args: Optional[str] = None,
    max_depth: int = 0,
    onnx_opset: Optional[int] = None,
    timeout: Optional[int] = None,
    sequence: Union[str, stage.Sequence] = None,
    rt_args: Optional[Dict] = None,
):
    # Capture the function arguments so that we can forward them
    # to downstream APIs
    benchmarking_args = copy.deepcopy(locals())
    regular_files = []

    # Replace .txt files with the models listed inside them
    input_files = unpack_txt_inputs(input_files)

    # Iterate through each string in the input_files list
    for input_string in input_files:
        if not any(char in input_string for char in "*?[]"):
            regular_files.append(input_string)

    # Create a list of files that don't exist on the filesystem
    # Skip the files with "::" as hashes will be decoded later
    non_existent_files = [
        file for file in regular_files if not os.path.exists(file) and "::" not in file
    ]
    if non_existent_files:
        raise exceptions.ArgError(
            f"{non_existent_files} do not exist, please verify if the file(s) exists."
        )

    # Make sure that `timeout` is only being used with `process_isolation` or `use_slurm`
    # And then set a default for timeout if the user didn't set a value
    if timeout is not None:
        if not use_slurm and not process_isolation:
            raise exceptions.ArgError(
                "The `timeout` argument is only allowed when slurm "
                "or process isolation mode is activated."
            )

        timeout_to_use = timeout
    else:
        timeout_to_use = spawn.DEFAULT_TIMEOUT_SECONDS

    benchmarking_args["timeout"] = timeout_to_use

    # Convert regular expressions in input files argument
    # into full file paths (e.g., [*.py] -> [a.py, b.py] )
    input_files_expanded = filesystem.expand_inputs(input_files)

    # Do not forward arguments to downstream APIs
    # that will be decoded in this function body
    benchmarking_args.pop("input_files")
    benchmarking_args.pop("labels")
    benchmarking_args.pop("use_slurm")
    benchmarking_args.pop("process_isolation")

    # Make sure the cache directory exists
    filesystem.make_cache_dir(cache_dir)

    check_sequence_type(sequence, use_slurm, process_isolation)

    if device is None:
        device = "x86"

    # Replace the runtime with a default value, if needed
    selected_runtime = devices.apply_default_runtime(device, runtime)
    benchmarking_args["runtime"] = selected_runtime

    # Get the default part and config by providing the Device class with
    # the supported devices by the runtime
    runtime_supported_devices = SUPPORTED_RUNTIMES[selected_runtime][
        "supported_devices"
    ]
    benchmarking_args["device"] = str(Device(device, runtime_supported_devices))

    # Force the user to specify a legal cache dir in NFS if they are using slurm
    if cache_dir == filesystem.DEFAULT_CACHE_DIR and use_slurm:
        printing.log_warning(
            "Using the default cache directory when using Slurm will cause your cached "
            "files to only be available at the Slurm node. If this is not the behavior "
            "you desired, please se a --cache-dir that is accessible by both the slurm "
            "node and your local machine."
        )

    # Get list containing only file names
    clean_file_names = [
        decode_input_arg(file_name)[0] for file_name in input_files_expanded
    ]

    # Validate that the files have supported file extensions
    # Note: We are not checking for .txt files here as those were previously handled
    for file_name in clean_file_names:
        if not file_name.endswith(".py") and not file_name.endswith(".onnx"):
            raise exceptions.ArgError(
                f"File extension must be .py, .onnx, or .txt (got {file_name})"
            )

    # Decode turnkey args into TracerArgs flags
    if analyze_only:
        actions = [
            Action.ANALYZE,
        ]
    elif build_only:
        actions = [
            Action.ANALYZE,
            Action.BUILD,
        ]
    else:
        actions = [
            Action.ANALYZE,
            Action.BUILD,
            Action.BENCHMARK,
        ]

    if use_slurm:
        jobs = spawn.slurm_jobs_in_queue()
        if len(jobs) > 0:
            printing.log_warning(f"There are already slurm jobs in your queue: {jobs}")
            printing.log_info(
                "Suggest quitting turnkey, running 'scancel -u $USER' and trying again."
            )

    # Use this data structure to keep a running index of all models
    models_found: Dict[str, ModelInfo] = {}

    # Fork the args for analysis since they have differences from the spawn args:
    # build_only and analyze_only are encoded into actions
    analysis_args = copy.deepcopy(benchmarking_args)
    analysis_args.pop("build_only")
    analysis_args.pop("analyze_only")
    analysis_args["actions"] = actions
    analysis_args.pop("timeout")

    for file_path_encoded in input_files_expanded:
        # Check runtime requirements if needed. All benchmarking will be halted
        # if requirements are not met. This happens regardless of whether
        # process-isolation is used or not.
        runtime_info = SUPPORTED_RUNTIMES[selected_runtime]
        if "requirement_check" in runtime_info and Action.BENCHMARK in actions:
            runtime_info["requirement_check"]()

        printing.log_info(f"Running turnkey on {file_path_encoded}")

        file_path_absolute, targets, encoded_input = decode_input_arg(file_path_encoded)

        # Skip a file if the required_labels are not a subset of the script_labels.
        if labels:
            # Labels argument is not supported for ONNX files
            if file_path_absolute.endswith(".onnx"):
                raise ValueError(
                    "The labels argument is not supported for .onnx files, got",
                    file_path_absolute,
                )
            required_labels = labels_library.to_dict(labels)
            script_labels = labels_library.load_from_file(encoded_input)
            if not labels_library.is_subset(required_labels, script_labels):
                continue

        if use_slurm or process_isolation:
            # Decode args into spawn.Target
            if use_slurm and process_isolation:
                raise ValueError(
                    "use_slurm and process_isolation are mutually exclusive, but both are True"
                )
            elif use_slurm:
                process_type = spawn.Target.SLURM
            elif process_isolation:
                process_type = spawn.Target.LOCAL_PROCESS
            else:
                raise ValueError(
                    "This code path requires use_slurm or use_process to be True, "
                    "but both are False"
                )

            spawn.run_turnkey(
                op="benchmark",
                target=process_type,
                file_name=encoded_input,
                **benchmarking_args,
            )

        else:
            # Instantiate an object that holds all of the arguments
            # for analysis, build, and benchmarking
            tracer_args = TracerArgs(
                models_found=models_found,
                targets=targets,
                input=file_path_absolute,
                **analysis_args,
            )

            if file_path_absolute.endswith(".py"):
                # Run analysis, build, and benchmarking on every model
                # in the python script
                models_found = evaluate_script(tracer_args)
            elif file_path_absolute.endswith(".onnx"):
                # Skip analysis and go straight to dealing with the model
                # We need to manufacture ModelInfo and UniqueInvocatioInfo instances to do this,
                # since we didn't get them from analysis.

                # Gather information about the ONNX model
                onnx_name = pathlib.Path(file_path_absolute).stem
                onnx_hash = get_model_hash(
                    file_path_absolute, build.ModelType.ONNX_FILE
                )
                onnx_inputs = onnx_helpers.dummy_inputs(file_path_absolute)
                input_shapes = {key: value.shape for key, value in onnx_inputs.items()}

                # Create the UniqueInvocationInfo
                #  - execute=1 is required or else the ONNX model will be
                #       skipped in later stages of evaluation
                #  - is_target=True is required or else traceback wont be printed for
                #       in the event of any errors
                #  - Most other values can be left as default
                invocation_info = UniqueInvocationInfo(
                    executed=1,
                    input_shapes=input_shapes,
                    hash=onnx_hash,
                    is_target=True,
                )

                # Create the ModelInfo
                model_info = ModelInfo(
                    model=file_path_absolute,
                    name=onnx_name,
                    script_name=onnx_name,
                    file=file_path_absolute,
                    build_model=not build_only,
                    model_type=build.ModelType.ONNX_FILE,
                    unique_invocations={onnx_hash: invocation_info},
                    hash=onnx_hash,
                )

                # Begin evaluating the ONNX model
                tracer_args.script_name = onnx_name
                tracer_args.models_found[tracer_args.script_name] = model_info
                explore_invocation(
                    model_inputs=onnx_inputs,
                    model_info=model_info,
                    invocation_info=invocation_info,
                    tracer_args=tracer_args,
                )
                models_found = tracer_args.models_found

    # Wait until all the Slurm jobs are done
    if use_slurm:
        while len(spawn.slurm_jobs_in_queue()) != 0:
            print(
                f"Waiting: {len(spawn.slurm_jobs_in_queue())} "
                f"jobs left in queue: {spawn.slurm_jobs_in_queue()}"
            )
            time.sleep(5)

    printing.log_success("The 'benchmark' command is complete.")
