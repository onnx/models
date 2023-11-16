import os
import numpy as np

import onnx
import onnxruntime
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType


class DataReader(CalibrationDataReader):
    """Wrapper class around calibration data, which is used to quantize an onnx model."""

    def __init__(self, input_file, samples, input_shapes=None, pack_inputs=False):
        session = onnxruntime.InferenceSession(input_file, None)
        input_names = [inp.name for inp in session.get_inputs()]

        if pack_inputs:
            expand_each = lambda data: [np.expand_dims(d, axis=0) for d in data]
            self.enum_data_dicts = iter(
                [
                    dict(zip(input_names, expand_each(sample_inputs)))
                    for sample_inputs in zip(*samples)
                ]
            )
        else:
            if input_shapes:
                self.samples = samples.reshape(-1, len(input_shapes), *input_shapes[0])
            else:
                self.samples = samples

            self.enum_data_dicts = iter(
                [dict(zip(input_names, sample)) for sample in self.samples]
            )

    def get_next(self):
        return next(self.enum_data_dicts, None)


def quantize(
    input_file,
    data,
    input_shapes=None,
    pack_inputs=False,
    verbose=False,
    output_file=None,
):
    """
    Given an onnx file and calibration data on which to quantize,
    computes and saves quantized onnx model to a local file.
    """
    data_reader = DataReader(
        input_file,
        samples=data,
        input_shapes=input_shapes,
        pack_inputs=pack_inputs,
    )

    if not output_file:
        output_file = input_file[:-5] + "_quantized.onnx"

    quantize_static(
        model_input=input_file,
        model_output=output_file,
        calibration_data_reader=data_reader,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv", "MatMul", "Relu"],
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )

    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(output_file)), output_file)

    if os.path.isfile("augmented_model.onnx"):
        os.remove("augmented_model.onnx")

    if verbose:
        print("Calibrated and quantized model saved.")

    return output_file
