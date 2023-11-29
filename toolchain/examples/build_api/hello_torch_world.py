"""
    Hello ** PyTorch ** World!

    This example uses a small model to carry out a single vector matrix
    multiplication to demonstrate building and build a PyTorch model.

    This example will help identify what you should expect from each build_model()
    PyTorch build. You can find the build results in the cache directory at
    ~/.cache/turnkey_test_cache/hello_pytorch_world/ (unless otherwise specified).
"""

import torch
from turnkeyml import build_model

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
inputs = {"x": torch.rand(input_size)}

# Build the model
state = build_model(
    pytorch_model,
    inputs,
    cache_dir="~/.cache/turnkey_test_cache",
)

# Print build results
print(f"Build status: {state.build_status}")
