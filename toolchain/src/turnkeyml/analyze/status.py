import os
import math
import platform
from typing import Dict, Union, List
from turnkeyml.common import printing
import turnkeyml.common.build as build
from turnkeyml.analyze.util import ModelInfo


def update(models_found: Dict[str, ModelInfo], build_name: str, cache_dir: str) -> None:
    """
    Prints all models and submodels found
    """
    if os.environ.get("TURNKEY_DEBUG") != "True":
        if platform.system() != "Windows":
            os.system("clear")
        else:
            os.system("cls")

    printing.logn(
        "\nModels discovered during profiling:\n",
        c=printing.Colors.BOLD,
    )
    recursive_print(models_found, build_name, cache_dir, None, None, [])


def recursive_print(
    models_found: Dict[str, ModelInfo],
    build_name: str,
    cache_dir: str,
    parent_model_hash: Union[str, None] = None,
    parent_invocation_hash: Union[str, None] = None,
    script_names_visited: List[str] = False,
) -> None:
    script_names_visited = []

    for model_hash in models_found.keys():
        model_visited = False
        model_info = models_found[model_hash]
        invocation_idx = 0
        for invocation_hash in model_info.unique_invocations.keys():
            unique_invocation = model_info.unique_invocations[invocation_hash]

            if (
                parent_model_hash == model_info.parent_hash
                and unique_invocation.executed > 0
                and (
                    model_info.unique_invocations[invocation_hash].parent_hash
                    == parent_invocation_hash
                )
            ):
                print_file_name = False
                if model_info.script_name not in script_names_visited:
                    script_names_visited.append(model_info.script_name)
                    if model_info.depth == 0:
                        print_file_name = True

                print_invocation(
                    model_info,
                    build_name,
                    cache_dir,
                    invocation_hash,
                    print_file_name,
                    invocation_idx=invocation_idx,
                    model_visited=model_visited,
                )
                model_visited = True
                invocation_idx += 1

                if print_file_name:
                    script_names_visited.append(model_info.script_name)

                recursive_print(
                    models_found,
                    build_name,
                    cache_dir,
                    parent_model_hash=model_hash,
                    parent_invocation_hash=invocation_hash,
                    script_names_visited=script_names_visited,
                )


def _pretty_print_key(key: str) -> str:
    result = key.split("_")
    result = [word.capitalize() for word in result]
    result = " ".join(result)
    return result


def parameters_to_size(parameters: int, byte_per_parameter: int = 4) -> str:
    size_bytes = parameters * byte_per_parameter
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def print_invocation(
    model_info: ModelInfo,
    build_name: str,
    cache_dir: str,
    invocation_hash: Union[str, None],
    print_file_name: bool = False,
    invocation_idx: int = 0,
    model_visited: bool = False,
) -> None:
    """
    Print information about a given model or submodel
    """
    unique_invocation = model_info.unique_invocations[invocation_hash]
    if model_info.model_type == build.ModelType.ONNX_FILE:
        extension = ".onnx"
        ident = "\t" * (2 * model_info.depth)
    else:
        extension = ".py"
        ident = "\t" * (2 * model_info.depth + 1)

    if print_file_name:
        print(f"{model_info.script_name}{extension}:")

    if unique_invocation.exec_time == 0 or model_info.build_model:
        exec_time = ""
    else:
        exec_time = f" - {unique_invocation.exec_time:.2f}s"

    # Print invocation about the model (only applies to scripts, not ONNX files)
    if model_info.model_type != build.ModelType.ONNX_FILE:
        if model_info.depth == 0 and len(model_info.unique_invocations) > 1:
            if not model_visited:
                printing.logn(f"{ident}{model_info.name}")
        else:
            printing.log(f"{ident}{model_info.name}")
            printing.logn(
                f" (executed {unique_invocation.executed}x{exec_time})",
                c=printing.Colors.OKGREEN,
            )

    if (model_info.depth == 0 and not model_visited) or (model_info.depth != 0):
        if model_info.depth == 0:
            if model_info.model_type == build.ModelType.PYTORCH:
                print(f"{ident}\tModel Type:\tPytorch (torch.nn.Module)")
            elif model_info.model_type == build.ModelType.KERAS:
                print(f"{ident}\tModel Type:\tKeras (tf.keras.Model)")
            elif model_info.model_type == build.ModelType.ONNX_FILE:
                print(f"{ident}\tModel Type:\tONNX File (.onnx)")

        # Display class of model and where it was found, if
        # the an input script (and not an input onnx file) was used
        if model_info.model_type != build.ModelType.ONNX_FILE:
            model_class = type(model_info.model)
            print(f"{ident}\tClass:\t\t{model_class.__name__} ({model_class})")
            if model_info.depth == 0:
                print(f"{ident}\tLocation:\t{model_info.file}, line {model_info.line}")

        # Display number of parameters and size
        parameters_size = parameters_to_size(model_info.params)
        print(
            f"{ident}\tParameters:\t{'{:,}'.format(model_info.params)} ({parameters_size})"
        )

    if model_info.depth == 0 and len(model_info.unique_invocations) > 1:
        printing.logn(
            f"\n{ident}\tWith input shape {invocation_idx+1} "
            f"(executed {unique_invocation.executed}x{exec_time})",
            c=printing.Colors.OKGREEN,
        )

    # Prepare input shape to be printed
    input_shape = dict(model_info.unique_invocations[invocation_hash].input_shapes)
    input_shape = {key: value for key, value in input_shape.items() if value != ()}
    input_shape = str(input_shape).replace("{", "").replace("}", "")

    print(f"{ident}\tInput Shape:\t{input_shape}")
    print(f"{ident}\tHash:\t\t" + invocation_hash)
    print(f"{ident}\tBuild dir:\t" + cache_dir + "/" + build_name)

    # Print turnkey results if turnkey was run
    if unique_invocation.performance:
        printing.log(f"{ident}\tStatus:\t\t")
        printing.logn(
            f"Successfully benchmarked on {unique_invocation.performance.device} "
            f"({unique_invocation.performance.runtime} "
            f"v{unique_invocation.performance.runtime_version}) ",
            c=unique_invocation.status_message_color,
        )
        printing.logn(
            f"{ident}\t\t\tMean Latency:\t{unique_invocation.performance.mean_latency:.3f}"
            f"\t{unique_invocation.performance.latency_units}"
        )
        printing.logn(
            f"{ident}\t\t\tThroughput:\t{unique_invocation.performance.throughput:.1f}"
            f"\t{unique_invocation.performance.throughput_units}"
        )

        if unique_invocation.stats_keys is not None:
            for key in unique_invocation.stats_keys:
                nice_key = _pretty_print_key(key)
                value = unique_invocation.stats.build_stats[key]
                printing.logn(f"{ident}\t\t\t{nice_key}:\t{value}")
        print()
    else:
        if unique_invocation.is_target and model_info.build_model:
            printing.log(f"{ident}\tStatus:\t\t")
            printing.logn(
                f"{unique_invocation.status_message}",
                c=unique_invocation.status_message_color,
            )

            if unique_invocation.traceback is not None:
                if os.environ.get("TURNKEY_TRACEBACK") != "False":
                    for line in unique_invocation.traceback:
                        for subline in line.split("\n")[:-1]:
                            print(f"{ident}\t{subline}")

                else:
                    printing.logn(
                        f"{ident}\t\t\tTo see the full stack trace, "
                        "rerun with `export TURNKEY_TRACEBACK=True`.\n",
                        c=model_info.status_message_color,
                    )
            else:
                print()
        print()
