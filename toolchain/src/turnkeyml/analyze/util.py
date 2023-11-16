import sys
from dataclasses import dataclass
from typing import Callable, List, Union, Dict, Optional
import dataclasses
import numpy as np
import torch
import onnx
from turnkeyml.common import printing
import turnkeyml.common.build as build
from turnkeyml.common.performance import MeasuredPerformance
from turnkeyml.common.filesystem import Stats


class AnalysisException(Exception):
    """
    Indicates a failure during analysis
    """


@dataclass
class UniqueInvocationInfo:
    """
    Refers to unique static model invocations
    (i.e. models executed with unique input shapes)
    """

    hash: Union[str, None] = None
    parent_hash: Union[str, None] = None
    performance: MeasuredPerformance = None
    traceback: List[str] = None
    inputs: Union[dict, None] = None
    input_shapes: Union[dict, None] = None
    executed: int = 0
    exec_time: float = 0.0
    status_message: str = ""
    is_target: bool = False
    status_message_color: printing.Colors = printing.Colors.ENDC
    traceback_message_color: printing.Colors = printing.Colors.FAIL
    stats_keys: Optional[List[str]] = None
    stats: Stats = None


@dataclass
class ModelInfo:
    model: torch.nn.Module
    name: str
    script_name: str
    file: str = ""
    line: int = 0
    params: int = 0
    depth: int = 0
    hash: Union[str, None] = None
    parent_hash: Union[str, None] = None
    old_forward: Union[Callable, None] = None
    unique_invocations: Union[
        Dict[str, UniqueInvocationInfo], None
    ] = dataclasses.field(default_factory=dict)
    last_unique_invocation_executed: Union[str, None] = None
    build_model: bool = False
    model_type: build.ModelType = build.ModelType.PYTORCH

    def __post_init__(self):
        self.params = count_parameters(self.model, self.model_type)


def count_parameters(model: torch.nn.Module, model_type: build.ModelType) -> int:
    """
    Returns the number of parameters of a given model
    """
    if model_type == build.ModelType.PYTORCH:
        return sum([parameter.numel() for _, parameter in model.named_parameters()])
    elif model_type == build.ModelType.KERAS:
        return model.count_params()
    elif model_type == build.ModelType.ONNX_FILE:
        onnx_model = onnx.load(model)
        return int(
            sum(
                np.prod(tensor.dims)
                for tensor in onnx_model.graph.initializer
                if tensor.name not in onnx_model.graph.input
            )
        )

    # Raise exception if an unsupported model type is provided
    raise AnalysisException(f"model_type {model_type} is not supported")


def get_onnx_ops_list(onnx_model) -> Dict:
    """
    List unique ops found in the onnx model
    """
    onnx_ops_counter = {}
    try:
        model = onnx.load(onnx_model)
    except Exception as e:  # pylint: disable=broad-except
        printing.log_warning(f"Failed to get ONNX ops list from {onnx_model}: {str(e)}")
        return onnx_ops_counter
    for node in model.graph.node:  # pylint: disable=E1101
        onnx_ops_counter[node.op_type] = onnx_ops_counter.get(node.op_type, 0) + 1
    return onnx_ops_counter


def populate_onnx_model_info(onnx_model) -> Dict:
    """
    Read the model metadata to populate IR, Opset and model size
    """
    result_dict = {
        "ir_version": None,
        "opset": None,
        "size on disk (KiB)": None,
    }
    try:
        model = onnx.load(onnx_model)
    except Exception as e:  # pylint: disable=broad-except
        printing.log_warning(f"Failed to get ONNX ops list from {onnx_model}: {str(e)}")
        result_dict.update({"error": "ONNX model analysis failed"})
        return result_dict
    # pylint: disable=E1101
    result_dict.update(
        {
            "ir_version": getattr(model, "ir_version", None),
            "opset": getattr(model.opset_import[0], "version", None),
        }
    )
    try:
        result_dict.update(
            {
                "size on disk (KiB)": round(
                    model.SerializeToString().__sizeof__() / 1024, 4
                ),
            }
        )
    except ValueError:
        # Models >2GB on disk cannot have their model size measured this
        # way and will throw a ValueError https://github.com/aig-bench/onnxmodelzoo/issues/318
        pass

    return result_dict


def onnx_input_dimensions(onnx_model) -> Dict:
    """
    Read model input dimensions
    """
    input_shape = {}
    try:
        model = onnx.load(onnx_model)
    except Exception as e:  # pylint: disable=broad-except
        printing.log_warning(f"Failed to get ONNX ops list from {onnx_model}: {str(e)}")
        return input_shape
    for inp in model.graph.input:  # pylint: disable=E1101
        shape = str(inp.type.tensor_type.shape.dim)
        input_shape[inp.name] = [int(s) for s in shape.split() if s.isdigit()]
    return input_shape


def stop_logger_forward() -> None:
    """
    Stop forwarding stdout and stderr to file
    """
    if hasattr(sys.stdout, "terminal"):
        sys.stdout = sys.stdout.terminal
    if hasattr(sys.stderr, "terminal_err"):
        sys.stderr = sys.stderr.terminal_err
