"""
    Hello ** ONNX ** World!

    This example uses a small model to carry out a single vector matrix
    multiplication to demonstrate building and running an ONNX model
    with build_model().

    This example will help identify what you should expect from each build_model()
    ONNX build. You can find the build results in the cache directory at
    ~/.cache/turnkey_test_cache/hello_onnx_world/ (unless otherwise specified).
"""

import os
import torch
from turnkeyml import build_model

torch.manual_seed(0)


# Start from a PyTorch model so you can generate an ONNX
# file to pass into build_model().
class SmallModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SmallModel, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        output = self.fc(x)
        return output


# Instantiate PyTorch model and generate inputs
input_size = 10
output_size = 5
pytorch_model = SmallModel(input_size, output_size)
onnx_model = "small_onnx_model.onnx"
input_tensor = torch.rand(input_size)
inputs = {"input": input_tensor}

# Export PyTorch Model to ONNX
torch.onnx.export(
    pytorch_model,
    input_tensor,
    onnx_model,
    opset_version=13,
    input_names=["input"],
    output_names=["output"],
)


# You can use numpy arrays as inputs to our ONNX model
def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


# Build the model
state = build_model(
    onnx_model,
    inputs,
    cache_dir="~/.cache/turnkey_test_cache",
)

# Remove intermediate onnx file so that you don't pollute your disk
if os.path.exists(onnx_model):
    os.remove(onnx_model)

# Print build results
print(f"Build status: {state.build_status}")
