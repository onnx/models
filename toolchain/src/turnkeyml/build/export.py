import os
import inspect
import shutil
import warnings
import sys
import copy
from typing import Union
import torch
import numpy as np
import onnxruntime
import onnxmltools
import onnx
import turnkeyml.build.stage as stage
import turnkeyml.common.exceptions as exp
import turnkeyml.common.build as build
import turnkeyml.build.tensor_helpers as tensor_helpers
import turnkeyml.build.onnx_helpers as onnx_helpers
import turnkeyml.build.quantization_helpers as quant_helpers
import turnkeyml.common.filesystem as fs


def check_model(onnx_file, success_message, fail_message) -> bool:
    if os.path.isfile(onnx_file):
        print(success_message)
    else:
        print(fail_message)
        return False
    try:
        onnx.checker.check_model(onnx_file)
        print("\tSuccessfully checked onnx file")
        return True
    except onnx.checker.ValidationError as e:
        print("\tError while checking generated ONNX file")
        print(e)
        return False


def _warn_to_stdout(message, category, filename, line_number, _, line):
    sys.stdout.write(
        warnings.formatwarning(message, category, filename, line_number, line)
    )


def get_output_names(
    onnx_model: Union[str, onnx.ModelProto]
):  # pylint: disable=no-member
    # Get output names of ONNX file/model
    if not isinstance(onnx_model, onnx.ModelProto):  # pylint: disable=no-member
        onnx_model = onnx.load(onnx_model)
    return [node.name for node in onnx_model.graph.output]  # pylint: disable=no-member


def onnx_dir(state: build.State):
    return os.path.join(
        build.output_dir(state.cache_dir, state.config.build_name), "onnx"
    )


def base_onnx_file(state: build.State):
    return os.path.join(
        onnx_dir(state),
        f"{state.config.build_name}-op{state.config.onnx_opset}-base.onnx",
    )


def opt_onnx_file(state: build.State):
    return os.path.join(
        onnx_dir(state),
        f"{state.config.build_name}-op{state.config.onnx_opset}-opt.onnx",
    )


def converted_onnx_file(state: build.State):
    return os.path.join(
        onnx_dir(state),
        f"{state.config.build_name}-op{state.config.onnx_opset}-opt-f16.onnx",
    )


def quantized_onnx_file(state: build.State):
    return os.path.join(
        onnx_dir(state),
        f"{state.config.build_name}-op{state.config.onnx_opset}-opt-quantized_int8.onnx",
    )


class ExportPlaceholder(stage.Stage):
    """
    Placeholder Stage that should be replaced by a framework-specific export stage,
    typically during ignition.model_intake()
    """

    def __init__(self):
        super().__init__(
            unique_name="export_placeholder",
            monitor_message="Placeholder for an Export Stage",
        )

    def fire(self, _: build.State):
        raise exp.StageError(
            "This Sequence includes an ExportPlaceholder Stage that should have "
            "been replaced with an export Stage."
        )


class ReceiveOnnxModel(stage.Stage):
    """
    Stage that takes an ONNX model as input.

    Expected inputs:
     - state.model is a path to the ONNX model
     - state.inputs is a dict that represents valid inputs for the onnx model

    Outputs:
     - A *-base.onnx file that implements state.model given state.inputs.
    """

    def __init__(self):
        super().__init__(
            unique_name="receive_onnx",
            monitor_message="Receiving ONNX Model",
        )

    def fire(self, state: build.State):
        if not isinstance(state.model, str):
            msg = f"""
            The current stage (ReceiveOnnxModel) is only compatible with
            ONNX files, however the stage received a model of type
            {type(state.model)}.
            """
            raise exp.StageError(msg)
        if not state.model.endswith(".onnx"):
            msg = f"""
            The current stage (ReceiveOnnxModel) expects a path to ONNX
            model, however the stage received {state.model}.
            """
            raise exp.StageError(msg)

        dummy_inputs = tuple(state.inputs.values())
        dummy_input_names = tuple(state.inputs.keys())
        state.inputs = dict(zip(dummy_input_names, dummy_inputs))

        model = onnx.load(state.model)
        opset = onnx_helpers.get_opset(model)
        input_shapes = [
            [d.dim_value for d in _input.type.tensor_type.shape.dim]
            for _input in model.graph.input  # pylint: disable=no-member
        ]

        # Save output node names
        state.expected_output_names = get_output_names(model)

        # Check for Dynamic shapes in the model. They can be represented as 0, -1, "unk__".
        for input in input_shapes:
            for dimension in input:
                if dimension < 1 or not isinstance(dimension, int):
                    msg = f"""
                    The received model has dynamic input dimensions. Please freeze the model with static
                    input dimensions.
                    More information may be available in the log file at **{self.logfile_path}**
                    """
                    raise exp.StageError(msg)

        if opset < build.DEFAULT_ONNX_OPSET and opset >= build.MINIMUM_ONNX_OPSET:
            print(
                f" \n The received model has an opset {opset}. Though this opset is supported \
                we recommend upgrading the model to opset {build.MINIMUM_ONNX_OPSET}"
            )
        elif opset < build.MINIMUM_ONNX_OPSET:
            msg = f"""
            The received model has an opset {opset}. Opset < 11 is not supported. Please
            try upgrading the model to opset 13.
            More information may be available in the log file at **{self.logfile_path}**
            """
            raise exp.StageError(msg)

        output_path = base_onnx_file(state)
        os.makedirs(onnx_dir(state), exist_ok=True)
        shutil.copy(state.model, output_path)

        tensor_helpers.save_inputs(
            [state.inputs], state.original_inputs_file, downcast=False
        )

        # Check the if the base mode has been exported successfully
        success_msg = "\tSuccess receiving ONNX Model"
        fail_msg = "\tFailed receiving ONNX Model"

        if check_model(output_path, success_msg, fail_msg):
            state.intermediate_results = [output_path]

            stats = fs.Stats(state.cache_dir, state.config.build_name, state.stats_id)
            stats.add_build_stat(
                fs.Keys.ONNX_FILE,
                output_path,
            )
        else:
            msg = f"""
            Unable to process ONNX Model. We recommend that you verify the source of the model.
            Any optimizations performed on the model could result in an error.
            More information may be available in the log file at **{self.logfile_path}**
            """
            raise exp.StageError(msg)

        return state


class ExportPytorchModel(stage.Stage):
    """
    Stage that takes a PyTorch model instance, in state.model, and
    exports it to an ONNX file.

    Expected inputs:
     - state.model is a torch.nn.Module or torch.jit.ScriptModule
     - state.inputs is a dict that represents valid kwargs to the forward
        function of state.model

    Outputs:
     - A *-base.onnx file that implements state.model given state.inputs
    """

    def __init__(self):
        super().__init__(
            unique_name="export_pytorch",
            monitor_message="Exporting PyTorch to ONNX",
        )

    def fire(self, state: build.State):
        if not isinstance(state.model, (torch.nn.Module, torch.jit.ScriptModule)):
            msg = f"""
            The current stage (ExportPytorchModel) is only compatible with
            models of type torch.nn.Module or torch.jit.ScriptModule, however
            the stage received a model of type {type(state.model)}.
            """
            raise exp.StageError(msg)

        # The `torch.onnx.export()` function accepts a tuple of positional inputs
        # followed by a dictionary with all keyword inputs.
        # The dictionary must be last item in tuple.
        user_provided_args = list(state.inputs.keys())

        if isinstance(state.model, torch.nn.Module):
            # Validate user provided args
            all_args = list(inspect.signature(state.model.forward).parameters.keys())

            for inp in user_provided_args:
                if inp not in all_args:
                    msg = f"""
                    Input name {inp} not found in the model's forward method. Available
                    input names are: {all_args}"
                    """
                    raise ValueError(msg)

            # Most pytorch models have args that are kind = positional_or_keyword.
            # The `torch.onnx.export()` function accepts model args as
            #     (all_positional_args_value,{keyword_arg:value}).
            # To map the input_args correctly and to build an accurate model
            # the order of the input_names must reflect the order of the model args.

            # Collect order of pytorch model args.
            all_args_order_mapping = {arg: idx for idx, arg in enumerate(all_args)}

            # Sort the user provided inputs with respect to model args and store as tuple.
            sorted_user_inputs = sorted(
                user_provided_args, key=lambda x: all_args_order_mapping[x]
            )
            dummy_input_names = tuple(sorted_user_inputs)

            # If a single input is provided torch.onnx.export will
            # not accept a dictionary, so pop the first arg
            user_args = copy.deepcopy(state.inputs)
            first_input = user_args.pop(dummy_input_names[0])

            # Create tuple: (first input, {rest of user_args dict as keyword args})
            dummy_inputs = (first_input, user_args)

        else:  # state.model is a torch.jit.ScriptModule
            dummy_inputs = tuple(state.inputs.values())

            # Collect input names
            dummy_input_names = tuple(state.inputs.keys())

        # Send torch export warnings to stdout (and therefore the log file)
        # so that they don't fill up the command line
        default_warnings = warnings.showwarning
        warnings.showwarning = _warn_to_stdout

        # Export the model to ONNX
        output_path = base_onnx_file(state)
        os.makedirs(onnx_dir(state), exist_ok=True)
        torch.onnx.export(
            state.model,
            dummy_inputs,
            output_path,
            input_names=dummy_input_names,
            do_constant_folding=True,
            opset_version=state.config.onnx_opset,
            verbose=False,
        )

        # Save output names to ensure we are preserving the order of the outputs
        state.expected_output_names = get_output_names(output_path)

        # Restore default warnings behavior
        warnings.showwarning = default_warnings

        tensor_helpers.save_inputs(
            [state.inputs], state.original_inputs_file, downcast=False
        )

        # Check the if the base mode has been exported successfully
        success_msg = "\tSuccess exporting model to ONNX"
        fail_msg = "\tFailed exporting model to ONNX"

        if check_model(output_path, success_msg, fail_msg):
            state.intermediate_results = [output_path]

            stats = fs.Stats(state.cache_dir, state.config.build_name, state.stats_id)
            stats.add_build_stat(
                fs.Keys.ONNX_FILE,
                output_path,
            )
        else:
            msg = f"""
            Unable to export model to ONNX using Torch's ONNX exporter.
            We recommend that you modify your model until it is
            compatible with this third party software, then re-run.
            More information may be available in the log file at **{self.logfile_path}**
            """
            raise exp.StageError(msg)

        return state


class ExportKerasModel(stage.Stage):
    """
    Stage that takes a Keras model instance, in state.model, and
    exports it to an ONNX file.

    Expected inputs:
     - state.model is a tf.keras.Model
     - state.inputs is a dict that represents valid kwargs to the forward
        function of state.model

    Outputs:
     - A *-base.onnx file that implements state.model given state.inputs
    """

    def __init__(self):
        super().__init__(
            unique_name="export_keras",
            monitor_message="Exporting Keras to ONNX",
        )

    def fire(self, state: build.State):
        # pylint: disable=import-error
        import tensorflow as tf
        import tf2onnx

        if not isinstance(state.model, (tf.keras.Model)):
            msg = f"""
            The current stage (ExportKerasModel) is only compatible with
            models of type tf.keras.Model, however
            the stage received a model of type {type(state.model)}.
            """
            raise exp.StageError(msg)

        user_provided_args = state.inputs.keys()

        all_args = []

        # Check the model inputs member
        if state.model.inputs:
            all_args = [x.name for x in state.model.inputs]

        # If the input name(s) cannot be extracted from the inputs variable
        # than try to find them in the call() method
        if len(all_args) == 0:
            all_args = list(inspect.signature(state.model.call).parameters.keys())

        inputs = []
        input_names = []

        for inp in user_provided_args:
            if inp not in all_args:
                msg = f"""
                Input name {inp} not found in the model's forward method. Available
                input names are: {all_args}"
                """
                raise ValueError(msg)

        for _, arg in enumerate(all_args):
            if arg in user_provided_args:
                inputs.append(state.inputs[arg])
                input_names.append(arg)

        input_specs = []
        for inp, name in zip(inputs, input_names):
            dtype = inp.dtype
            shape = inp.shape
            if inp.dtype == tf.float64:
                print(f"Converting input {name} from float64 to float32")
                dtype = tf.float32
            if inp.dtype == tf.int64:
                print(f"Converting input {name} from int64 to int32")
                dtype = tf.int32
            if inp.shape[0] is None:
                print("Found batch size None and setting it to 1")
                shape = (1, shape[1:])

            input_specs.append(tf.TensorSpec(shape, dtype, name))

        # Export the model to ONNX
        output_path = base_onnx_file(state)
        os.makedirs(onnx_dir(state), exist_ok=True)
        tf2onnx.convert.from_keras(
            state.model,
            input_signature=input_specs,
            opset=state.config.onnx_opset,
            output_path=output_path,
        )

        # Save output names to ensure we are preserving the order of the outputs
        state.expected_output_names = get_output_names(output_path)

        state.inputs = dict(zip(tuple(input_names), tuple(inputs)))

        tensor_helpers.save_inputs(
            [state.inputs], state.original_inputs_file, downcast=False
        )

        # Check the if the base mode has been exported successfully
        success_msg = "\tSuccess exporting model to ONNX"
        fail_msg = "\tFailed exporting model to ONNX"

        if check_model(output_path, success_msg, fail_msg):
            state.intermediate_results = [output_path]

            stats = fs.Stats(state.cache_dir, state.config.build_name, state.stats_id)
            stats.add_build_stat(
                fs.Keys.ONNX_FILE,
                output_path,
            )
        else:
            msg = f"""
            Unable to export model to ONNX using tf2onnx exporter.
            We recommend that you modify your model until it is
            compatible with this third party software, then re-run.
            More information may be available in the log file at **{self.logfile_path}**
            """
            raise exp.StageError(msg)

        return state


class OptimizeOnnxModel(stage.Stage):
    """
    Stage that takes an ONNX file and uses ONNX Runtime to optimize it.
    Important because this helps to perform constant folding, Redundant
    node eliminations, Semantics-preserving node fusions

    Expected inputs:
     - state.intermediate_results contains a single .onnx file

    Outputs:
     - A *-opt.onnx file
    """

    def __init__(self):
        super().__init__(
            unique_name="optimize_onnx",
            monitor_message="Optimizing ONNX file",
        )

    def fire(self, state: build.State):
        input_onnx = state.intermediate_results[0]
        output_path = opt_onnx_file(state)

        # Perform some basic optimizations on the model to remove shape related
        # information inserted for dynamic shape inference.
        # Given that we're compiling against a fixed sequence length the dynamic
        # shape information is not necessary
        session_options = onnxruntime.SessionOptions()

        # Set graph optimization level
        session_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )

        # To enable model serialization after graph optimization set this
        session_options.optimized_model_filepath = output_path

        # Optimize graph
        onnxruntime.InferenceSession(input_onnx, session_options)

        # Check that the converted model is still valid
        success_msg = "\tSuccess optimizing ONNX model"
        fail_msg = "\tFailed optimizing ONNX model"

        if check_model(output_path, success_msg, fail_msg):
            state.intermediate_results = [output_path]

            stats = fs.Stats(state.cache_dir, state.config.build_name, state.stats_id)
            stats.add_build_stat(
                fs.Keys.ONNX_FILE,
                output_path,
            )
        else:
            msg = f"""
            Unable to optimize ONNX file using ONNX runtime.
            We recommend that you modify your model until it is
            compatible with this third party software, then re-run.
            More information may be available in the log file at **{self.logfile_path}**
            """
            raise exp.StageError(msg)

        return state


class ConvertOnnxToFp16(stage.Stage):
    """
    Stage that takes an ONNX file and converts its trained parameters
    to fp16.

    Expected inputs:
     - state.intermediate_results contains a single .onnx file

    Outputs:
     - A *-f16.onnx file with FP16 trained parameters
    """

    def __init__(self):
        super().__init__(
            unique_name="fp16_conversion",
            monitor_message="Converting to FP16",
        )

    def fire(self, state: build.State):
        input_onnx = state.intermediate_results[0]

        # Convert the model to FP16
        # Some ops will not be converted to fp16 because they are in a block list
        # The latest list can be found here. It is not necessarily the list that
        # our version of onnxmltools sees
        # https://github.com/microsoft/onnxconverter-common/blob/master/onnxconverter_common/float16.py#L82

        # Send onnxmltools warnings to stdout (and therefore the log file)
        # so that they don't fill up the command line
        default_warnings = warnings.showwarning
        warnings.showwarning = _warn_to_stdout

        # Legalize ops are ops that have been or are currently in the block list
        # that we explicitly want removed
        legalize_ops = ["InstanceNormalization", "Resize", "Max"]
        op_block_list = onnxmltools.utils.float16_converter.DEFAULT_OP_BLOCK_LIST.copy()
        for op in legalize_ops:
            # Check to see that they are not in the block list before we remove them
            # Neccesary because the block list may be updated, and not in the state we expect
            if op in op_block_list:
                op_block_list.remove(op)

        # Infer shapes before converting to FP16 to enable models with >2GB
        onnx.shape_inference.infer_shapes_path(input_onnx)

        fp32_model = onnx.load_model(input_onnx)
        fp16_model = onnxmltools.utils.float16_converter.convert_float_to_float16(
            fp32_model, op_block_list=op_block_list, disable_shape_infer=True
        )

        # Load inputs and convert to fp16
        inputs_file = state.original_inputs_file
        if os.path.isfile(inputs_file):
            inputs = np.load(inputs_file, allow_pickle=True)
            to_downcast = False if state.quantization_samples else True
            inputs_converted = tensor_helpers.save_inputs(
                inputs, inputs_file, downcast=to_downcast
            )
        else:
            raise exp.StageError(
                "Attempted to convert inputs to FP16, however inputs file was not found."
            )

        # Overwrite expected dtypes
        _, state.expected_input_dtypes = build.get_shapes_and_dtypes(
            inputs_converted[0]
        )

        # Indicate that inputs must be downcasted during inference
        state.downcast_applied = True

        # Save FP16 model (use external data format if needed)
        output_path = converted_onnx_file(state)
        try:
            onnxmltools.utils.save_model(fp16_model, output_path)
        except ValueError:
            onnx.save_model(fp16_model, output_path, save_as_external_data=True)

        # Restore default warnings behavior
        warnings.showwarning = default_warnings

        # Check that the converted model is still valid
        success_msg = "\tSuccess converting ONNX model to fp16"
        fail_msg = "\tFailed converting ONNX model to fp16"

        if check_model(output_path, success_msg, fail_msg):
            state.intermediate_results = [output_path]

            stats = fs.Stats(state.cache_dir, state.config.build_name, state.stats_id)
            stats.add_build_stat(
                fs.Keys.ONNX_FILE,
                output_path,
            )
        else:
            msg = f"""
            Attempted to use onnxmltools, a third party library, to convert your
            model to the float16 datatype, however this operation was not successful.
            More information may be available in the log file at **{self.logfile_path}**
            """
            raise exp.StageError(msg)

        return state


class QuantizeONNXModel(stage.Stage):
    """
    Stage that takes an ONNX model and a dataset of quantization samples as inputs,
    and performs static post-training quantization to the model to int8 precision.

    Expected inputs:
     - state.model is a path to the ONNX model
     - state.quantization_dataset is a dataset that is used for static quantization

    Outputs:
     - A *_quantized.onnx file => the quantized onnx model.
    """

    def __init__(self):
        super().__init__(
            unique_name="quantize_onnx",
            monitor_message="Quantizing ONNX model",
        )

    def fire(self, state: build.State):
        input_path = state.intermediate_results[0]
        output_path = quantized_onnx_file(state)

        quant_helpers.quantize(
            input_file=input_path,
            data=state.quantization_samples,
            output_file=output_path,
        )

        # Check that the converted model is still valid
        success_msg = "\tSuccess quantizing ONNX model to int8"
        fail_msg = "\tFailed quantizing ONNX model to int8"

        if check_model(output_path, success_msg, fail_msg):
            state.intermediate_results = [output_path]

            stats = fs.Stats(state.cache_dir, state.config.build_name, state.stats_id)
            stats.add_build_stat(
                fs.Keys.ONNX_FILE,
                output_path,
            )
        else:
            msg = f"""
            Attempted to use {state.quantization_dataset} to statically quantize
            model to int8 datatype, however this operation was not successful.
            More information may be available in the log file at **{self.logfile_path}**
            """
            raise exp.StageError(msg)

        return state


class SuccessStage(stage.Stage):
    """
    Stage that sets state.build_status = build.Status.SUCCESSFUL_BUILD,
    indicating that the build sequence has completed all of the requested build stages.
    """

    def __init__(self):
        super().__init__(
            unique_name="set_success",
            monitor_message="Finishing up",
        )

    def fire(self, state: build.State):
        state.build_status = build.Status.SUCCESSFUL_BUILD

        state.results = copy.deepcopy(state.intermediate_results)

        return state
