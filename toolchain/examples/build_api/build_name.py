"""
    This example demonstrates changing the directory name within the cache directory
    (~/.cache/turnkey) where all the logs, artifacts, and the state file will be written.

    To change the directory name, use the build_name argument with a unique name.

    The directory for each build defaults to the name of the file it was built in;
    'build_name' would be the default for this file.

    Note: If a single script is used to build multiple models, (or if a build_name
    matches a build directory within cache already), then a unique build_name will
    need to be defined, or the subsequent build(s) will overwrite (or load) the
    previous build found in ~/.cache/turnkey/{non_unique_build_name}.
    See docs/tools_user_guide.md for more information.
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


# Create two different model instances, each with a different output
# size. You can check the build artifacts to verify that both models
# are built and stored separately.
input_size = 10
output_size_1 = 5
output_size_2 = 8

pytorch_model_1 = SmallModel(input_size, output_size_1)
pytorch_model_2 = SmallModel(input_size, output_size_2)
inputs = {"x": torch.rand(input_size)}

# Build pytorch_model_1 and write build files to ~/.cache/turnkey/Thing_1
build_model(pytorch_model_1, inputs, build_name="Thing_1")

# Build pytorch_model_2 and write build files to ~/.cache/turnkey/Thing_2
build_model(pytorch_model_2, inputs, build_name="Thing_2")

print("\nNote that each build is saved to their own build directories")
print("as indicated at the completion of each build above.")

print("Example build_name.py finished")
