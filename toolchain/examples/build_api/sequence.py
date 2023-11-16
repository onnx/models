"""

    By default, build_model() completes the following steps:
     > Convert to ONNX
     > Optimize ONNX file
     > Convert to FP16
     > Finish up

    This example illustrates how to alter the default sequence of steps. In this
    example, the conversion to FP16 is skipped.
"""

import torch
from turnkeyml import build_model
import turnkeyml.build.export as export
import turnkeyml.build.stage as stage


torch.manual_seed(0)


# Define model class
class SmallModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SmallModel, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        output = self.fc(x)
        return output


# Instantiate model and generate inputs
input_size = 10
output_size = 5

pytorch_model = SmallModel(input_size, output_size)
inputs = {"x": torch.rand(input_size, dtype=torch.float32)}

onnx_sequence = stage.Sequence(
    "onnx_sequence",
    "Building ONNX Model without fp16 conversion",
    [
        export.ExportPytorchModel(),
        export.OptimizeOnnxModel(),
        # export.ConvertOnnxToFp16(),  #<-- This is the step we want to skip
        export.SuccessStage(),
    ],
    enable_model_validation=True,
)

# Build model
build_model(pytorch_model, inputs, sequence=onnx_sequence)
